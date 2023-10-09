# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import random
import sys
from functools import partial
from typing import Iterable

import torch
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import utils
from flow_utils.softsplat import BackWarp, Softsplat


def train_one_epoch(model: torch.nn.Module,
                    flow_model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    args,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    build_mask_volume = get_build_mask_volume_func(flow_model, device, args)

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # NOTE: In MGMAE, the init_mask_map is the mask map of the base frame,
        # not the entire clip. If decoder mask type is none, decode_mask_pos
        # will be ~mask_volume after the mask_volume was built.
        images, init_mask_map, decode_masked_pos = batch

        images = images.to(device, non_blocking=True)
        init_mask_map = init_mask_map.unsqueeze(1).to(
            device, non_blocking=True).to(torch.float32)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                     None,
                                                                     None,
                                                                     None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                   None, None,
                                                                   None]
            unnorm_images = images * std + mean  # in [0, 1]
            nimgs = 2 * unnorm_images - 1.0  # in [-1, 1]
            mask_volume = build_mask_volume(nimgs, init_mask_map)
            if args.decoder_mask_type == 'none':
                decode_masked_pos = ~mask_volume

            if normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)

            B, N, C = images_patch.shape
            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)

        if loss_scaler is None:
            outputs = model(images, mask_volume, decode_masked_pos)
            loss = (outputs - labels)**2
            loss = loss.mean(dim=-1)
            cal_loss_mask = mask_volume[~decode_masked_pos].reshape(B, -1)
            loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(images, mask_volume, decode_masked_pos)
                loss = (outputs - labels)**2
                loss = loss.mean(dim=-1)
                cal_loss_mask = mask_volume[~decode_masked_pos].reshape(B, -1)
                loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()

        if loss_scaler is None:
            loss.backward()
            if max_norm is None:
                grad_norm = utils.get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()
            loss_scale_value = 0
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_build_mask_volume_func(flow_model, device, args):
    if args.warp_type == 'backward':
        # The method of warping flow on the mask map.
        warpFlow = partial(BackWarp, device=device)
    if args.warp_type == 'forward':
        B = args.batch_size * args.num_sample
        H = W = args.input_size
        z = torch.oes(B, 1, H, W).to(device, non_blocking=True)
        warpFlow = Softsplat(z=z)

    def _build_mask_volume(nimgs, init_mask_map):
        B, C, T, H, W = nimgs.shape
        base_idx = {
            'middle': T // 2,
            'first': 0,
            'random': random.randint(0, T - 1)
        }.get(args.base_frame, T // 2)

        mask_map_list = [init_mask_map]

        mask_map = init_mask_map
        for r_idx in reversed(range(base_idx)):
            # determine the order of frames to do flow warpping.
            l_idx = r_idx + 1
            if args.warp_type == 'backwarp':
                l_idx, r_idx = r_idx, l_idx
            _, flow = flow_model(
                nimgs[:, :, l_idx],
                nimgs[:, :, r_idx],
                iters=args.flow_iter,
                test_mode=True)
            mask_map = warpFlow(mask_map, flow)
            if args.hole_filling == 'consist':
                empty_pos = (mask_map == 0)
                mask_map[empty_pos] = init_mask_map[empty_pos]
            mask_map_list.insert(0, mask_map)

        mask_map = init_mask_map
        for l_idx in range(base_idx, T - 1):
            r_idx = l_idx + 1
            if args.warp_type == 'backward':
                # Exchange frames order in backwarp warpping
                l_idx, r_idx = r_idx, l_idx
            _, flow = flow_model(
                nimgs[:, :, l_idx],
                nimgs[:, :, r_idx],
                iters=args.flow_iter,
                test_mode=True)
            mask_map = warpFlow(mask_map, flow)
            if args.hole_filling == 'consist':
                empty_pos = (mask_map == 0)
                mask_map[empty_pos] = init_mask_map[empty_pos]
            mask_map_list.append(mask_map)

        # mask_map_list list(B, C(1), H, W)
        mask_map = torch.cat(mask_map_list, dim=1)
        # [B, t, 1]
        mask_map = rearrange(
            mask_map,
            'b (t p0) (h p1) (w p2) -> b t (h w) (p0 p1 p2)',
            t=args.window_size[0],
            h=args.window_size[1],
            w=args.window_size[2]).sum(-1)

        if args.exposure_chance:
            B, T, L = mask_map.shape
            selected_frame = torch.randint(0, T, (B, ))
            noise_mask = torch.arange(T).unsqueeze(0).expand(B, T)
            noise_mask = (noise_mask == selected_frame.unsqueeze(1))
            noise = torch.randn(B, 1, L).expand(
                B, T, L) * noise_mask.unsqueeze(-1).float()
            mask_map = mask_map + noise.to(device)

        num_vis = int(args.window_size[1] * args.window_size[2] *
                      (1 - args.mask_ratio))
        if args.topk_on_all:
            mask_map = mask_map.flatten(1)
            num_vis = num_vis * args.window_size[0]

        vis_index = torch.topk(
            mask_map, num_vis, dim=-1, largest=True, sorted=False).indices
        mask_volume = torch.ones_like(mask_map).scatter_(-1, vis_index, 0)
        mask_volume = mask_volume.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        return mask_volume

    return _build_mask_volume
