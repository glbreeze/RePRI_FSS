## modified the code: args.batch_size_val == 1


import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from visdom_logger import VisdomLogger
from collections import defaultdict
from .dataset.dataset import get_val_loader
from .util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, main_process
from .util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from .classifier import Classifier
from .model.pspnet import get_model
import torch.distributed as dist
from tqdm import tqdm
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from .visu import make_episode_visualization
from typing import Tuple


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:

    print(f"==> Running DDP checkpoint example on rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        np.random.seed(args.manual_seed + rank)
        torch.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)
        random.seed(args.manual_seed + rank)

    # ========== Model  ==========
    model = get_model(args).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    root = get_model_dir(args)

    if args.ckpt_used is not None:
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    # ========== Test  ==========
    val_Iou = episodic_validate(args=args,
                                          val_loader=episodic_val_loader,
                                          model=model,
                                          use_callback=(args.visdom_port != -1),
                                          suffix=f'test')
    if args.distributed:
        dist.all_reduce(val_Iou)
        val_Iou /= world_size

    cleanup()


def episodic_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    nb_episodes = int(args.test_num)

    # ========== Metrics initialization  ==========

    H, W = args.image_size, args.image_size
    c = model.module.bottleneck_dim
    if args.image_size == 473:
        h, w = 60, 60
    else:
        h, w = model.module.feature_res   # (53, 53)

    runtimes = torch.zeros(args.n_runs)
    deltas_init = torch.zeros((args.n_runs, nb_episodes, 1))
    deltas_final = torch.zeros((args.n_runs, nb_episodes, 1))
    val_IoUs0 = np.zeros(args.n_runs)

    # ========== Perform the runs  ==========
    for run in tqdm(range(args.n_runs)):

        # =============== Initialize the metric dictionaries ===============

        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection0 = defaultdict(int)  # Default value is 0
        cls_union0 = defaultdict(int)
        IoU0 = defaultdict(int)

        cls_intersection1 = defaultdict(int)  # Default value is 0
        cls_union1 = defaultdict(int)
        IoU1 = defaultdict(int)

        # =============== episode = group of tasks ===============
        runtime = 0
        for e in tqdm(range(nb_episodes)):
            t0 = time.time()
            iter_num += 1

            try:
                q_img, q_label, s_img, s_label, subcls, _, _ = iter_loader.next()
            except:
                iter_loader = iter(val_loader)
                q_img, q_label, s_img, s_label, subcls, _, _ = iter_loader.next()
            iter_num += 1
            q_img = q_img.to(dist.get_rank(), non_blocking=True)      # [1, 3, h, w]
            q_label = q_label.to(dist.get_rank(), non_blocking=True)  # [1, h, w]
            s_img = s_img.to(dist.get_rank(), non_blocking=True)      # [1, 1, 3, h, w]
            s_label = s_label.to(dist.get_rank(), non_blocking=True)  # [1, 1, h, w]

            classes = [[class_.item() for class_ in subcls]]  # All classes considered in the tasks, list of list

            def fit_model(s_img_, s_label_, q_img_, q_label_, classes):
                features_s = torch.zeros(1, args.shot * args.meta_aug, c, h, w).to(dist.get_rank())  # [1, 2, c, h, w]
                features_q = torch.zeros(1, 1, c, h, w).to(dist.get_rank())                          # [1, 1, c, h, w]
                gt_s = 255 * torch.ones(1, args.shot * args.meta_aug, args.image_size, args.image_size).long().to(dist.get_rank())  # [1, 2, h, w]
                gt_q = 255 * torch.ones(1, 1, args.image_size, args.image_size).long().to(dist.get_rank())                          # [1, 1, h, w]
                n_shots = torch.zeros(1).to(dist.get_rank())

                # =========== Generate tasks and extract features for each task ===============
                with torch.no_grad():
                    f_s = model.module.extract_features(s_img_.squeeze(0))    # [shot, ch, h, w]
                    f_q = model.module.extract_features(q_img_)               # [1, ch, h, w]

                    shot = f_s.size(0)
                    n_shots[0] = shot
                    features_s[0, :shot] = f_s.detach()
                    features_q[0] = f_q.detach()
                    gt_s[0, :shot] = s_label_   # [1, 2/shot, h, w]   ====
                    gt_q[i, 0] = q_label_       # [1, h, w]           ====

                # =========== Normalize features along channel dimension ===============
                if args.norm_feat:
                    features_s = F.normalize(features_s, dim=2)
                    features_q = F.normalize(features_q, dim=2)

                # =========== Create a callback is args.visdom_port != -1 ===============
                callback = VisdomLogger(port=args.visdom_port) if use_callback else None

                # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
                classifier = Classifier(args)
                classifier.init_prototypes(features_s, features_q, gt_s, gt_q, classes, callback)
                batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
                deltas_init[run, e, :] = batch_deltas.cpu()

                # =========== Perform RePRI inference ===============
                batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback)
                deltas_final[run, e, :] = batch_deltas

                logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                probas = classifier.get_probas(logits).detach()
                intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
                intersection, union = intersection.cpu(), union.cpu()
                return intersection, union

            intersection0, union0 = fit_model(s_img[:,:1], s_label[:, :1], q_img, q_label, classes)   # s_img [1, 1, 3, h, w] s_label # [1, 1, h, w]
            intersection1, union1 = fit_model(s_img, s_label, q_img, q_label, classes)
            runtime += time.time() - t0
            # ================== Log metrics ==================

            for i, task_classes in enumerate(classes):
                for j, class_ in enumerate(task_classes):
                    cls_intersection0[class_] += intersection0[i, 0, j + 1]  # Do not count background
                    cls_union0[class_] += union0[i, 0, j + 1]
                    cls_intersection1[class_] += intersection1[i, 0, j + 1]  # Do not count background
                    cls_union1[class_] += union1[i, 0, j + 1]

            for class_ in cls_union0:
                IoU0[class_] = cls_intersection0[class_] / (cls_union0[class_] + 1e-10)
                IoU1[class_] = cls_intersection1[class_] / (cls_union1[class_] + 1e-10)

            if (iter_num % 200 == 0):
                mIoU0 = np.mean([IoU0[i] for i in IoU0])
                mIoU1 = np.mean([IoU1[i] for i in IoU1])
                print('Test: [{}/{}] '
                      'mIoU0 {:.4f}, mIoU1 {:.4f}'.format(iter_num, args.test_num, mIoU0, mIoU1))

            # ================== Visualization ==================

        runtimes[run] = runtime
        mIoU0 = np.mean(list(IoU0.values()))
        mIoU1 = np.mean(list(IoU1.values()))
        print('mIoU---Val result: mIoU0 {:.4f}, mIoU1 {:.4f}.'.format(mIoU0, mIoU1))
        for class_ in cls_union0:
            print("Class {} : IoU0 {:.4f}, IoU1 {:.4f}".format(class_, IoU0[class_], IoU1[class_]))

        val_IoUs0[run] = mIoU0

    # ================== Save metrics ==================
    if args.save_oracle:
        root = os.path.join('plots', 'oracle')
        os.makedirs(root, exist_ok=True)
        np.save(os.path.join(root, 'delta_init.npy'), deltas_init.numpy())
        np.save(os.path.join(root, 'delta_final.npy'), deltas_final.numpy())

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs0.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs0.mean()


def standard_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Standard validation')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    iterable_val_loader = iter(val_loader)

    bar = tqdm(range(len(iterable_val_loader)))

    loss = 0.
    intersections = torch.zeros(args.num_classes_tr).to(dist.get_rank())
    unions = torch.zeros(args.num_classes_tr).to(dist.get_rank())

    with torch.no_grad():
        for i in bar:
            images, gt = iterable_val_loader.next()
            images = images.to(dist.get_rank(), non_blocking=True)
            gt = gt.to(dist.get_rank(), non_blocking=True)
            logits = model(images).detach()
            loss += loss_fn(logits, gt)
            intersection, union, _ = intersectionAndUnionGPU(logits.argmax(1),
                                                             gt,
                                                             args.num_classes_tr,
                                                             255)
            intersections += intersection
            unions += union
        loss /= len(val_loader.dataset)

    if args.distributed:
        dist.all_reduce(loss)
        dist.all_reduce(intersections)
        dist.all_reduce(unions)

    mIoU = (intersections / (unions + 1e-10)).mean()
    loss /= dist.get_world_size()
    return mIoU, loss


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)