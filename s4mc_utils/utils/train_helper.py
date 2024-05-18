import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from s4mc_utils.dataset.builder import get_loader
from s4mc_utils.models.model_helper import ModelBuilder
from s4mc_utils.utils.dist_helper import setup_distributed
from s4mc_utils.utils.loss_helper import get_criterion
from s4mc_utils.utils.lr_helper import get_optimizer, get_scheduler
from s4mc_utils.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed,
)


def train_semi(
        model,
        model_teacher,
        optimizer,
        lr_scheduler,
        sup_loss_fn,
        loader_l,
        loader_u,
        epoch,
        tb_logger,
        logger,
        memobank,
        queue_ptrlis,
        queue_size,
        cfg,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]
    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    B_ratio = len(loader_l) / len(loader_u)

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _ = loader_u_iter.next()
        image_u = image_u.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                            model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                    "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            indicator = cfg["trainer"]["unsupervised"].get("indicator", "margin")
            neigborhood_size = cfg["trainer"]["unsupervised"].get("neigborhood_size", 4)
            n_neigbors = cfg["trainer"]["unsupervised"].get("n_neigbors", 1)
            ds = "pascal" if "pascal" in cfg["dataset"]["type"] else "cityscapes"

            unsup_loss = compute_unsupervised_loss(
                pred_u_large,
                label_u_aug.clone(),
                drop_percent,
                pred_u_large_teacher.detach(),
                indicator,
                ds,
                neigborhood_size,
                n_neigbors) * cfg["trainer"]["unsupervised"].get("unsup_weight", 1) * B_ratio

            # contrastive for using S4MC+U2PL
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                        1 - epoch / cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                            entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                            entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                    .float()
                                    .unsqueeze(1)
                                    .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                if cfg_contra.get("binary", False):
                    contra_flag += " BCE"
                    contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    if not cfg_contra.get("anchor_ema", False):
                        new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                        )
                    else:
                        prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )

                dist.all_reduce(contra_loss)
                contra_loss = contra_loss / world_size

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss  # * 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                            i_iter
                            - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                            + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]  "
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})  "
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})  "
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})  "
                "u2pl {con_loss.avg:.3f} "
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    'S4MC',
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)

    return


def save_for_heatmap(model, data_loader, i):
    model.eval()
    data_loader.sampler.set_epoch(0)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank = dist.get_rank()
    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        output = outs["pred"]
        output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
        # output = output.data.max(1)[1]#.cpu().numpy()

        if step == 0:
            _output = output
        else:
            _output = torch.cat((_output, output), 0)
        if step == 5:
            break

    dist.all_reduce(output)
    torch.save(_output.cpu(), 'imgs/hm/' + str(i) + 'output.pt')
    return


def validate(
        model,
        data_loader,
        epoch,
        logger,
        cfg,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()  # shape

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )
        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


def compare(base_model, model, data_loader, logger):
    base_model.eval()
    model.eval()
    data_loader.sampler.set_epoch(0)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)
            base_out = base_model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
        output = output.data.max(1)[1]  # .cpu().numpy()

        base_output = base_out["pred"]
        base_output = F.interpolate(base_output, labels.shape[1:], mode="bilinear", align_corners=True)
        base_output = base_output.data.max(1)[1]  # .cpu().numpy()

        labels[labels == 255] = 0
        if step == 0:
            _images = images
            _labels = labels
            _output = output
            _base_output = base_output
        else:
            _images = torch.cat((_images, images), 0)
            _labels = torch.cat((_labels, labels), 0)
            _output = torch.cat((_output, output), 0)
            _base_output = torch.cat((_base_output, base_output), 0)
        # if step==4:
        #    break

    torch.save(_images.cpu(), 'imgs/images_' + str(rank) + '.pt')
    torch.save(_labels.cpu(), 'imgs/labels_' + str(rank) + '.pt')
    torch.save(_output.cpu(), 'imgs/output_' + str(rank) + '.pt')
    torch.save(_base_output.cpu(), 'imgs/base_output_' + str(rank) + '.pt')

    dist.all_reduce(labels)

    if rank == 0:
        logger.info(" *done comparing* ")
    return
