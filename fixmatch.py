import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def retern_expected_neighbourhoods(cfg):
    pascal_E_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
     0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
       0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
        0.9948415228771397, 0.993315147276186]
    cityscapes_E_joit= [0.9964431323326034, 0.9861639854327645, 0.961200350864672, 0.831981383888851,
     0.9733805255682632, 0.9763628214927057, 0.9914859783638212, 0.9967060462154962, 0.9877697391019615,
     0.98401636296503, 0.9905921230630785, 0.990570758079667, 0.9847896602829518, 0.9842767938040367,
     0.9516618302498175,0.9894203625002091,0.9713330344039811,0.9448128823732177,0.9544919165416704]
    E_joint = pascal_E_joint if cfg['dataset']=='pascal' else cityscapes_E_joit
    return E_joint


def main():
    passed_per = []
    passed_acr = []
    save_every = 1
    counter = 0

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    
    t = cfg['conf_thresh']
    
    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s = 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        
        
        if epoch % save_every == 0:
          if len(passed_per) > 0:
            passed_per[-1] /= counter
            passed_acr[-1] /= counter
            
            counter = 0
          

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
                (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            
            #parameters
            beta = 0.4 #torch.exp(torch.tensor(-1/2))
            E_joint = retern_expected_neighbourhoods(cfg)

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
            pred_u_s = model(img_u_s)
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= t) & (ignore_mask_cutmixed != 255))
            loss_u_s = torch.sum(loss_u_s) / torch.sum(ignore_mask_cutmixed != 255).item()

            loss = (loss_x + loss_u_s) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += loss_u_s.item()
            total_mask_ratio += ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss s: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_x / (i+1), total_loss_s / (i+1), total_mask_ratio / (i+1)))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))



if __name__ == '__main__':
    main()

