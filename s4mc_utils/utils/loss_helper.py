import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import dequeue_and_enqueue


def compute_rce_loss(predict, target):
    from einops import rearrange

    predict = F.softmax(predict, dim=1)

    with torch.no_grad():
        _, num_cls, h, w = predict.shape
        temp_tar = target.clone()
        temp_tar[target == 255] = 0

        label = (
            F.one_hot(temp_tar.clone().detach(), num_cls).float().cuda()
        )  # (batch, h, w, num_cls)
        label = rearrange(label, "b h w c -> b c h w")
        label = torch.clamp(label, min=1e-4, max=1.0)

    rce = -torch.sum(predict * torch.log(label), dim=1) * (target != 255).bool()
    return rce.sum() / (target != 255).sum()



def get_neigbor_tensors(X: torch.tensor, n: int = 4, entrophy=False):
    """
    Args:
        X: tensor of shape (B, C, H, W)
        n: number of neighbors to get 4 or 8
    Returns:
        list of tensors of shape (B, C, H, W)
    """
    assert n in [4, 8]
    if entrophy:
        X = X.unsqueeze(1)
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0)).squeeze(1)
    else:
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0))
    
    neigbors=[]
    if n==8:
        for i,j in [(None,-2),(1,-1),(2,None)]:
            for k,l in [(None,-2),(1,-1),(2,None)]:
                if i==k and i==1:
                    continue
                if entrophy:
                    neigbors.append(X[:,i:j, k:l])
                else:
                    neigbors.append(X[:,:,i:j, k:l])
    elif n ==4:
        if entrophy:
            neigbors.append(X[:,:-2, 1:-1])
            neigbors.append(X[:,2:,1:-1])
            neigbors.append(X[:,1:-1,:-2])
            neigbors.append(X[:,1:-1, 2:])    
        else:
            neigbors.append(X[:,:,:-2, 1:-1])
            neigbors.append(X[:,:,2:,1:-1])
            neigbors.append(X[:,:,1:-1,:-2])
            neigbors.append(X[:,:,1:-1, 2:])
    return neigbors



def compute_unsupervised_loss(predict, target, percent, pred_teacher,indicator='margin',ds='pascal',
                              neigborhood_size=4,n_neigbors=1):
    """
    Compute the unsupervised loss for the student network
    Args:
        predict: the prediction of the student model (after interpolation)
        target: pseudo labels from the teacher model (after augmentation from 'generate_unsup_data') 
        percent: the percentage of dropped unrelable data points (0.8->1, should be the percent we leave in)
        pred_teacher: the prediction of the teacher model (after interpolation)
        indicator: the indicator to drop pixels, can be 'entropy', 'confidence', 'margin'
        ds: the dataset, can be 'pascal' or 'cityscapes'
    Returns:
        the unsupervised loss
    """
    
    pascal_E_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
     0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
       0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
        0.9948415228771397, 0.993315147276186]
    cityscapes_E_joit= [0.9964431323326034, 0.9861639854327645, 0.961200350864672, 0.831981383888851,
     0.9733805255682632, 0.9763628214927057, 0.9914859783638212, 0.9967060462154962, 0.9877697391019615,
     0.98401636296503, 0.9905921230630785, 0.990570758079667, 0.9847896602829518, 0.9842767938040367,
     0.9516618302498175,0.9894203625002091,0.9713330344039811,0.9448128823732177,0.9544919165416704]
    E_joint = pascal_E_joint if ds=='pascal' else cityscapes_E_joit


    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        prob = torch.softmax(pred_teacher, dim=1) #shape: batch_size, num_class, h, w
        
        if indicator=='entropy':
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1) #shape: batch_size, h, w
            thresh = np.percentile(
                entropy[target != 255].detach().cpu().numpy().flatten(), percent)
            
        elif indicator=='confidence':
            _max_probs, max_class = torch.max(prob, dim=1) #shape: batch_size, h, w

            thresh = np.percentile(
                _max_probs[target != 255].detach().cpu().numpy().flatten(), 100-percent)
        else: #margin
            _topk,_=torch.topk(prob, 2, dim=1)
            _margin = _topk[:,0,:,:] - _topk[:,1,:,:] #shape: batch_size, h, w
            thresh = np.percentile(_margin[target != 255].detach().cpu().numpy().flatten(), 100-percent)

        E= torch.tensor(E_joint).view((1, num_class, 1, 1)).to(prob.device)
        """
        max_neigbor=True
        if max_neigbor:
            neighbors=torch.stack(get_neigbor_tensors(prob,n=neigborhood_size,entrophy=False))
            neighbor,neigbor_idx=torch.max(neighbors,0) 
            
            prob = prob + beta*neighbor - (torch.max(prob*neighbor,prob*E)*beta)
        """
        
        neighbors=torch.stack(get_neigbor_tensors(prob,n=neigborhood_size,entrophy=False))
        k_neighbors,neigbor_idx=torch.topk(neighbors, k=n_neigbors,axis=0)
        for neighbor in k_neighbors:
            beta = torch.exp(torch.tensor(-1/2)) #for more neigbors use neigbor_idx
            prob = prob + beta*neighbor - (torch.max(prob*neighbor,prob*E)*beta)
                

        
        if indicator=='entropy':
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)        
            thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()
        elif indicator=='confidence':
            max_probs, _ = torch.max(prob, dim=1)
            thresh_mask = max_probs.le(thresh).bool() * (target != 255).bool()
        elif indicator=='margin':
            _topk,_=torch.topk(prob, 2, dim=1)
            margin = _topk[:,0,:,:] - _topk[:,1,:,:]
            thresh_mask = margin.le(thresh).bool() * (target != 255).bool()
        
        target[thresh_mask] = 255
        total_pixels = batch_size * h * w
        total_pseudo = torch.sum(target != 255)
        scale_weight = total_pixels / total_pseudo

        
    loss = scale_weight * F.cross_entropy(predict, target, ignore_index=255)

    return loss


def compute_contra_memobank_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    cfg,
    memobank,
    queue_prtlis,
    queue_size,
    rep_teacher,
    momentum_prototype=None,
    i_iter=0,
):

    """if momentum_prototype is None:
        return 0,0
    else:
        return 0,0,0"""

    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg["current_class_threshold"]
    current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
    temp = cfg["temperature"]
    num_queries = cfg["num_queries"]
    num_negatives = cfg["num_negatives"]

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1]

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(
        0, 2, 3, 1
    )  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        rep_mask_low_entropy = (
            prob_seg > current_class_threshold
        ) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (
            prob_seg < current_class_negative_threshold
        ) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
                and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                high_entropy_idx = torch.randint(
                    len(negative_feat), size=(num_queries * num_negatives,)
                )
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )
                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .cuda()
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg


def get_criterion(cfg,cons=False):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    if cons:
        gamma = cfg['criterion']['cons']['gamma']
        sample = cfg['criterion']['cons'].get('sample', False)
        gamma2 = cfg['criterion']['cons'].get('gamma2',1)
        criterion = Criterion_cons(gamma, sample=sample, gamma2=gamma2, 
                                ignore_index=ignore_index, **cfg_criterion['kwargs'])
    return criterion


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss



class Criterion_cons(nn.Module):
    def __init__(self, gamma, sample=False, gamma2=1, 
                ignore_index=255, thresh=0.7, min_kept=100000, use_weight=False):
        super(Criterion_cons, self).__init__()
        self.gamma = gamma
        self.gamma2 = float(gamma2)
        self._ignore_index = ignore_index
        self.sample = sample
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def forward(self, preds, conf, gt, dcp_criterion=None):

        #conf = F.softmax(conf)
        ce_loss = self._criterion(preds,gt)
        conf = torch.pow(conf,self.gamma)

        if self.sample:
            dcp_criterion = 1 - dcp_criterion
            dcp_criterion = dcp_criterion / (torch.max(dcp_criterion)+1e-12) 
            dcp_criterion = torch.pow(dcp_criterion, self.gamma2)
            pred_map = preds.max(1)[1].float()

            sample_map = torch.zeros_like(pred_map).float()
            h, w = pred_map.shape[-2], pred_map.shape[-1]

            for idx in range(len(dcp_criterion)):
                prob = 1 - dcp_criterion[idx]
                rand_map = torch.rand(h, w).cuda()*(pred_map == idx)
                rand_map = (rand_map>prob)*1.0
                sample_map += rand_map
            conf = conf * (sample_map)
        conf = conf/(conf.sum() + 1e-12)

        loss = conf * ce_loss
        return loss.sum()

class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
