import argparse
import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from s4mc_utils.dataset.augmentation import generate_unsup_data
from s4mc_utils.dataset.builder import get_loader
from s4mc_utils.models.model_helper import ModelBuilder
from s4mc_utils.utils.dist_helper import setup_distributed, do_all_gather
from s4mc_utils.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from s4mc_utils.utils.visual_utils import visual_evaluation
from s4mc_utils.utils.lr_helper import get_optimizer, get_scheduler
from s4mc_utils.utils.boundary_utils import get_expectation, mask_to_boundary, neighbor_by_factor
from s4mc_utils.utils.train_helper import validate, compare, train_semi as train
from s4mc_utils.utils.utils import (
    AverageMeter,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

# for AEL run
# from s4mc_utils.utils.utils import (dynamic_copy_paste,update_cutmix_bank,generate_cutmix_mask,cal_category_confidence,sample_from_bank)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--name", default="regular")
parser.add_argument("--mode", type=str, default="train")


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        tb_logger = SummaryWriter(osp.join(cfg["exp_path"], f"log/events_seg/{args.name}_{seed}"))
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # If you run with u2pl or ael:
    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    q = cfg["trainer"]["contrastive"]["num_queries"]
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros((cfg["net"]["num_classes"], q, 1, 256,)).cuda()

    # Start to train model
    if cfg["main_mode"].get("eval", False):
        ## TODO: add this selection to configuration
        # uncomment this line if you want to use the refine_infer function
        # prec = refine_infer(model_teacher, val_loader, 0, logger)
        prec = validate(model_teacher, val_loader, 0, logger, cfg)
        if rank == 0:
            logger.info("evaluation result: {}".format(prec))
    elif cfg["main_mode"].get("compare", False):
        # Teacher model
        base_model = ModelBuilder(cfg["net"])
        base_model = base_model.cuda()
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        for p in base_model.parameters():
            p.requires_grad = False

        load_state(cfg["main_mode"]["compared_pretrain"], base_model, key="teacher_state")
        if rank == 0:
            logger.info("start comparison")
        compare(base_model, model_teacher, val_loader, logger)
    else:
        for epoch in range(last_epoch, cfg_trainer["epochs"]):
            # Training
            train(
                model,
                model_teacher,
                optimizer,
                lr_scheduler,
                sup_loss_fn,
                train_loader_sup,
                train_loader_unsup,
                epoch,
                tb_logger,
                logger,
                memobank,
                queue_ptrlis,
                queue_size,
                cfg)

            # Validation
            if cfg_trainer["eval_on"]:
                if rank == 0:
                    logger.info("start evaluation")

                if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                    prec = validate(model, val_loader, epoch, logger, cfg)
                else:
                    prec = validate(model_teacher, val_loader, epoch, logger, cfg)

                if rank == 0:
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "teacher_state": model_teacher.state_dict(),
                        "best_miou": best_prec,
                    }
                    if not os.path.exists(osp.join(cfg["saver"]["snapshot_dir"], args.name)):
                        os.makedirs(osp.join(cfg["saver"]["snapshot_dir"], args.name))

                    if prec > best_prec:
                        best_prec = prec
                        torch.save(
                            state, osp.join(osp.join(cfg["saver"]["snapshot_dir"], args.name), "ckpt_best.pth")
                        )

                    torch.save(state, osp.join(osp.join(cfg["saver"]["snapshot_dir"], args.name), "ckpt.pth"))

                    logger.info(
                        "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                            best_prec * 100
                        )
                    )
                    tb_logger.add_scalar("mIoU val", prec, epoch)


def refine_infer(
        model,
        data_loader,
        epoch,
        logger,
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

        prob = F.softmax(output.data, dim=1)

        arr, neigbor_idx = neighbor_by_factor(prob, 1)
        beta = torch.exp(torch.tensor(-1 / 2))

        arr2, neigbor_idx = neighbor_by_factor(2)
        arr3, neigbor_idx = neighbor_by_factor(4)
        arr4, neigbor_idx = neighbor_by_factor(5)

        prob = prob + beta * arr - (prob * arr * beta)
        # prob = prob + beta*arr2 - (prob*arr2*beta)
        # prob = prob + beta*arr3 - (prob*arr3*beta)
        prob = prob + beta * arr4 - (prob * arr4 * beta)

        output = prob.max(1)[1].cpu().numpy()

        # output = output.data.max(1)[1].cpu().numpy()

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


def sort_array_by_array(array, array_to_sort):
    return array[np.argsort(array_to_sort)]


if __name__ == "__main__":
    main()
