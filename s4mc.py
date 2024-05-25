import argparse
from ast import arg
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
from util.s4mc_utils import get_neigbor_tensors, expected_neighbourhoods

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--per', default=0.4, type=float)
parser.add_argument('--s4mc', default=None, type=int)
parser.add_argument('--fixmatch', default=None, type=int)


def load_model(cfg, rank, local_rank, logger, save_path=""):
    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    if os.path.exists(os.path.join(save_path, 'latest.pth')):
        if rank == 0:
            logger.info('load from last failed point')
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    return model, optimizer


def get_dataloader(trainset, cfg):
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    return trainloader, trainsampler


def main():
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

    local_rank = int(os.environ["LOCAL_RANK"])
    model, optimizer = load_model(cfg, rank, logger)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    ## data preparation
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainloader_l, trainsampler_l = get_dataloader(trainset_l)
    trainloader_u, trainsampler_u = get_dataloader(trainset_u)
    valloader, valsampler = get_dataloader(valset)


    # DPA parameter
    p_min = cfg["data_percentage"]
    p = p_min
    torch.cuda.empty_cache()

    # S4MC: Our parameters
    n = cfg["neighbourhood_size"]
    k = cfg["neighbours"]

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        if rank == 0:
            logger.info("quantile: " + str(p) + " ; step = " + str((1 - p_min) / cfg["epochs"]))

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask_u),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()
            # optimizer.zero_grad() #moshe: added for memory efficient
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # moshe: return if bottom does not work
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]  # feature perturbation

            # moshe: check for memory!
            pred_x, _ = model(img_x, True)
            loss_x = criterion_l(pred_x, mask_x) / 2
            torch.distributed.barrier()
            # loss_x.backward()
            pred_u_w, pred_u_w_fp = model(img_u_w, True)

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            # parameters
            beta = 1.0 if args.s4mc else 0.0  # torch.exp(torch.tensor(-1/2))
            e_joint = expected_neighbourhoods(cfg)

            prob = conf_u_w

            neighbors = torch.stack(get_neigbor_tensors(prob, n=n, entrophy=False))
            k_neighbors, neigbor_idx = torch.topk(neighbors, k=k, axis=0)
            k_neighbors = k_neighbors.to(conf_u_w.device)
            for neighbor in k_neighbors:
                beta = torch.exp(torch.tensor(-1 / 2))
                prob = prob + beta * neighbor - (prob * neighbor * beta)

            conf_u_w = prob.max(dim=1)[0]

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)

            conf, pred = conf_u_w_cutmixed1, mask_u_w_cutmixed1
            p = 1 if p > 1 else p
            t = np.quantile(conf.clone().detach().cpu().numpy(), 1 - p)

            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= t) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= t) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = torch.sum(loss_u_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()
            if args.fixmatch or args.s4mc:
                loss_u_w_fp = torch.zeros(1).to(loss_x.device)
            else:
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= t) & (ignore_mask != 255))
                loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            if args.fixmatch:
                loss = (loss_x + loss_u_s1 * 1. + loss_u_s2 * 1. + loss_u_w_fp * 0.0) / 2.0  # fixmatch
            elif args.s4mc:
                loss = (loss_x + loss_u_s1 * 0.5 + loss_u_s2 * 0.5 + loss_u_w_fp * 0.0) / 2.0
            else:
                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0  # regular

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += (loss_u_s1.item() + loss_u_s2.item()) / 2.0
            total_loss_w_fp += loss_u_w_fp.item()
            total_mask_ratio += ((conf_u_w >= t) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i + 1), total_loss_x / (i + 1), total_loss_s / (i + 1),
                       total_loss_w_fp / (i + 1), total_mask_ratio / (i + 1)))

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

        p += (0.85 - p_min) / cfg['epochs']  # linear increase to 1 at run end


if __name__ == '__main__':
    main()
