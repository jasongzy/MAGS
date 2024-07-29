#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, l1_loss_weighted, l2_loss_weighted
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer

import lpips
from utils.scene_utils import render_training_image
from utils.flow_utils import CosineAnnealing, pixel_to_gaussian_idx, project, query_feat_map, render_flow, get_neighbor_cams

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scene_reconstruction(dataset: ModelParams, opt: OptimizationParams, hyper: ModelHiddenParams, pipe: PipelineParams,
                        testing_iterations: "list[int]", saving_iterations: "list[int]", checkpoint_iterations: "list[int]", checkpoint: str, debug_from: int,
                        gaussians: GaussianModel, scene: Scene, stage: str, tb_writer, train_iter: int, timer: Timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        info = os.path.basename(checkpoint).replace(".pth", "").split("_")
        ckpt_stage, ckpt_iter = info[:2]
        if ckpt_stage == "best":
            ckpt_stage = "fine"
        assert ckpt_stage in ("coarse", "fine")
        if ckpt_stage == stage:
            print(f"Loading checkpoint: {checkpoint}")
            (model_params, first_iter) = torch.load(checkpoint)
            # assert first_iter == int(ckpt_iter)
            first_iter = int(ckpt_iter)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    if opt.lambda_lpips !=0:
        lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda()
    progress_bar = tqdm(range(first_iter, final_iter), desc=f"Training-{stage}", dynamic_ncols=True)
    first_iter += 1
    video_cams = scene.getVideoCameras()
    weight_scheduler = CosineAnnealing(warmup_step=opt.flow_warmup_step, total_step=final_iter, max_value=1.0, min_value=0.0)
    weight_scheduler.enable = opt.flow_weight_decay
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            batch_size = 1
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=opt.num_workers,collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader")
                batch_size = 1
                loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(loader)
            idx = viewpoint_cams[0].uid
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            viewpoint_cams = [viewpoint_stack[idx]]
            assert viewpoint_cams[0].uid == idx
        viewpoint_cams: "list[Camera]"

        next_cam, prev_cam = viewpoint_cams[0].next_cam, viewpoint_cams[0].prev_cam
        if opt.use_flow:
            assert next_cam or prev_cam
        extra_cams = [c for c in (next_cam, prev_cam) if c is not None]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, extra_cams=extra_cams)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor)
        # Ll1 = l2_loss(image, gt_image)
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm

        loss = torch.zeros_like(Ll1)
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor.to(image_tensor))
            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        if opt.lambda_lpips != 0:
            lpipsloss = lpips_loss(image_tensor, gt_image_tensor.to(image_tensor), lpips_model)
            loss += opt.lambda_lpips * lpipsloss

        lambda_flow = opt.lambda_flow * weight_scheduler.get_value(iteration)
        if opt.use_flow and stage == "fine" and next_cam is not None and (float(psnr_) >= opt.flow_psnr_threshold or opt.flow_render):
            # import time
            # tic = time.time()
            # assert len(viewpoint_cams) == 1
            viewpoint_cam = viewpoint_cams[0]
            flow, flow_bwd, occ, occ_bwd = viewpoint_cam.flow
            flow, flow_bwd, occ, occ_bwd = (d.cuda() if d is not None else None for d in (flow, flow_bwd, occ, occ_bwd))
            if occ is None:
                occ = torch.zeros([H, W]).bool().cuda()
            if occ_bwd is None:
                occ_bwd =  torch.zeros([H, W]).bool().cuda()
            w2c = viewpoint_cam.w2c.cuda()
            k = viewpoint_cam.k.cuda()

            is_fg = torch.ones_like(gaussians.get_opacity).bool().squeeze(-1)
            render_fn = lambda flow2render: render(
                viewpoint_camera=viewpoint_cam,
                pc=gaussians,
                pipe=pipe,
                bg_color=background,
                stage=stage,
                override_color=flow2render,
            )
            if opt.flow_render:
                fg_gaussian_idx = is_fg.clone()
                # seg_map = ~(occ | occ_bwd)
            else:
                # torch.cuda.empty_cache()
                depth = render_pkg["depth"]
                # depth_plot = depth.permute(1, 2, 0).detach().cpu().numpy()
                # depth_plot = (depth_plot - depth_plot.min()) / (depth_plot.max() - depth_plot.min())
                seg_map = (torch.rand_like(depth).squeeze(0) <= opt.flow_sample_ratio)  # randomly sample part of of pixels
                # seg_map = torch.zeros_like(depth).squeeze(0).bool()
                # seg_map[seg_map.shape[0]//2-200:seg_map.shape[0]//2+200, seg_map.shape[1]//2-200:seg_map.shape[1]//2+200] = True
                seg_map &= ~(occ | occ_bwd)
                if not seg_map.any():
                    seg_map[seg_map.shape[0] // 2, seg_map.shape[1] // 2] = True

                # # Test unproject & project
                # from utils.flow_utils import unproject
                # pts_unproject = unproject(depth, w2c, k)[:, seg_map].reshape(3, -1).T.detach()
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pts_unproject.cpu().numpy())
                # o3d.io.write_point_cloud("test.ply", pcd)
                # # uv_project, _ = project(pts_unproject.T, w2c, k)
                # # uv_project = uv_project.T.detach().cpu().numpy()
                # # import matplotlib.pyplot as plt
                # # fig = plt.figure()
                # # ax = fig.add_subplot(111)
                # # ax.scatter(uv_project[:, 0], uv_project[:, 1], s=1)
                # # ax.set_xlim([0, seg_map.shape[1]])
                # # ax.set_ylim([0, seg_map.shape[0]])
                # # ax.invert_yaxis()
                # # ax.set_aspect(1)
                # # plt.savefig("test.png")
                if opt.soft_select_fg:
                    # assert not opt.flow_render
                    if "gs_per_pixel" in render_pkg:
                        fg_gaussian_idx = (render_pkg["gs_per_pixel"][:, seg_map]).T
                        fg_gaussian_contrib = (render_pkg["weight_per_gs_pixel"][:, seg_map]).T.unsqueeze(-1)
                    else:
                        fg_gaussian_idx, fg_gaussian_contrib = pixel_to_gaussian_idx(
                            depth, w2c, k, seg_map, gaussians, is_fg, K=opt.flow_k, return_K=True
                        )
                    fg_shape = fg_gaussian_idx.shape
                    fg_gaussian_contrib = torch.nan_to_num(fg_gaussian_contrib, posinf=torch.finfo(fg_gaussian_contrib.dtype).max)
                    # fg_gaussian_contrib = fg_gaussian_contrib / (fg_gaussian_contrib.max(1, keepdim=True).values + 1e-8)
                    fg_gaussian_contrib_max = fg_gaussian_contrib.max(1, keepdim=True).values
                    # fg_gaussian_contrib_max[fg_gaussian_contrib_max < 1.0] = torch.inf  # if all K contributions are too small, set them to 0
                    fg_gaussian_contrib = fg_gaussian_contrib / (fg_gaussian_contrib_max + 1e-8)
                    fg_gaussian_idx = fg_gaussian_idx.reshape(-1)
                    if hyper.predict_confidence:
                        confidence = render_pkg["confidence"][fg_gaussian_idx].reshape(*fg_shape, 1)
                        fg_gaussian_contrib_ = fg_gaussian_contrib
                        fg_gaussian_contrib = fg_gaussian_contrib * confidence
                    flow_loss_fn = lambda pred, gt: l2_loss_weighted(
                        pred.reshape(*fg_shape, 2),
                        gt.unsqueeze(1).repeat_interleave(fg_shape[-1], dim=1),
                        weight=fg_gaussian_contrib,
                    ) + ((-torch.log(confidence + 1e-10) * fg_gaussian_contrib_).mean() if hyper.predict_confidence else 0.0)
                else:
                    if "gs_per_pixel" in render_pkg:
                        fg_gaussian_idx = render_pkg["gs_per_pixel"][:, seg_map].T
                        fg_gaussian_top_idx = (render_pkg["weight_per_gs_pixel"][:, seg_map]).T.argmax(-1).unsqueeze(-1)
                        fg_gaussian_idx = torch.gather(fg_gaussian_idx, dim=1, index=fg_gaussian_top_idx).squeeze(-1)
                    else:
                        fg_gaussian_idx = pixel_to_gaussian_idx(depth, w2c, k, seg_map, gaussians, is_fg, K=opt.flow_k, return_K=False)
                    # fg_gaussian_idx = torch.unique(fg_gaussian_idx)
                    # flow_loss_fn = l1_loss
                    flow_loss_fn = F.smooth_l1_loss

            # assert torch.all(gaussians._deformation_table)
            if flow is not None and next_cam is not None:
                time_next = torch.tensor(next_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
                means3D_deform_next = gaussians._deformation(
                    gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, time_next
                )[0]
                means3D_deform_next = means3D_deform_next[fg_gaussian_idx]
                if "gs_per_pixel" in render_pkg:
                    uv0 = render_pkg["proj_means_2D"][fg_gaussian_idx].T
                    if hyper.predict_flow:
                        means3D_deform = render_pkg["means3D_deform"][fg_gaussian_idx]
                else:
                    means3D_deform = render_pkg["means3D_deform"][fg_gaussian_idx]
                    uv0, _ = project(means3D_deform.T, w2c, k)
                uv1, _ = project(means3D_deform_next.T, next_cam.w2c.cuda(), next_cam.k.cuda())
                # uv0_int = torch.clamp(
                #     uv0.T.clone(),
                #     max=torch.tensor([W - 1, H - 1]).to(uv0),
                # ).round()
                # uv0_int = torch.clamp(uv0_int, min=0)
                # uv0_int = uv0_int.long()

                # # Test Gaussian projection
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(means3D_deform.detach().cpu().numpy())
                # o3d.io.write_point_cloud("test2.ply", pcd)
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # uv0_plot = uv0.T.detach().cpu().numpy()
                # uv1_plot = uv1.T.detach().cpu().numpy()
                # # ax.imshow(viewpoint_cam.original_image.detach().cpu().numpy().transpose(1, 2, 0))
                # ax.imshow(render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
                # ax.scatter(uv0_plot[:, 0], uv0_plot[:, 1], s=0.1, c="b")
                # ax.scatter(uv1_plot[:, 0], uv1_plot[:, 1], s=0.1, c="r")
                # ax.set_xlim([0, seg_map.shape[1]])
                # ax.set_ylim([0, seg_map.shape[0]])
                # ax.invert_yaxis()
                # ax.set_aspect(1)
                # plt.savefig("test.png")

                flow_project = (uv1 - uv0).T
                if opt.flow_render:
                    flow_render = render_flow(torch.clamp(flow_project, min=-W, max=W), render_fn=render_fn)
                    flow_loss = flow_loss_fn(flow_render.permute(1, 2, 0)[~occ], flow[~occ])
                    # import matplotlib.pyplot as plt
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # uv1_plot = (uv0.T + flow_render.permute(1, 2, 0)[uv0_int[:, 1], uv0_int[:, 0]]).detach().cpu().numpy()
                    # ax.scatter(uv0_plot[:, 0], uv0_plot[:, 1], s=0.1, c="b")
                    # ax.scatter(uv1_plot[:, 0], uv1_plot[:, 1], s=0.1, c="r")
                    # ax.set_xlim([0, seg_map.shape[1]])
                    # ax.set_ylim([0, seg_map.shape[0]])
                    # ax.invert_yaxis()
                    # ax.set_aspect(1)
                    # plt.savefig("test.png")
                else:
                    if opt.soft_select_fg:
                        uv0_int = torch.stack(torch.where(seg_map.T), dim=-1)
                        flow_fwd_uv0 = flow[uv0_int[:, 1], uv0_int[:, 0]]  # N,2
                    else:
                        flow_fwd_uv0 = query_feat_map(uv0.T, flow)  # N,2
                    flow_loss = flow_loss_fn(flow_project, flow_fwd_uv0)
                loss += lambda_flow * flow_loss

                if flow_bwd is not None and prev_cam is not None:
                    time_prev = torch.tensor(prev_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
                    means3D_deform_prev = gaussians._deformation(
                        gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, time_prev
                    )[0]
                    means3D_deform_prev = means3D_deform_prev[fg_gaussian_idx]
                    uv00, _ = project(means3D_deform_prev.T, prev_cam.w2c.cuda(), prev_cam.k.cuda())
                    flow_bwd_project = (uv00 - uv0).T
                    if opt.flow_render:
                        flow_bwd_render = render_flow(torch.clamp(flow_bwd_project, min=-W, max=W), render_fn=render_fn)
                        flow_bwd_loss = flow_loss_fn(flow_bwd_render.permute(1, 2, 0)[~occ_bwd], flow[~occ_bwd])
                    else:
                        if opt.soft_select_fg:
                            flow_bwd_uv0 = flow_bwd[uv0_int[:, 1], uv0_int[:, 0]]  # N,2
                        else:
                            flow_bwd_uv0 = query_feat_map(uv0.T, flow_bwd)  # N,2
                        flow_bwd_loss = flow_loss_fn(flow_bwd_project, flow_bwd_uv0)
                    loss += lambda_flow * flow_bwd_loss

                if hyper.predict_flow:
                    # fg_gaussian_idx = is_fg.clone()
                    velocity = render_pkg["velocity"][fg_gaussian_idx]
                    dt = next_cam.time - viewpoint_cam.time
                    uv0_pred = uv0
                    # uv0_pred = project(render_pkg["means3D_deform"][fg_gaussian_idx].T, w2c, k)[0]
                    uv1_pred, _ = project((means3D_deform + velocity * dt).T, next_cam.w2c.cuda(), next_cam.k.cuda())
                    flow_project_pred = (uv1_pred - uv0_pred).T
                    if opt.flow_render:
                        # flow_project_pred = torch.clamp(flow_project_pred, min=-W // 2, max=W // 2)
                        flow_render_pred = render_flow(flow_project_pred, render_fn=render_fn)
                        flow_pred_loss = flow_loss_fn(flow_render_pred.permute(1, 2, 0)[~occ], flow[~occ])
                    else:
                        flow_pred_loss = flow_loss_fn(flow_project_pred, flow_fwd_uv0)
                    loss += lambda_flow * 0.1 * flow_pred_loss

                    if opt.dynamic_attn and float(psnr_) >= opt.flow_psnr_threshold * 1.2:
                        # dynamic_map = torch.norm(flow, dim=-1)
                        with torch.no_grad():
                            # v_project, _ = project(render_pkg["velocity"].T * dt, w2c, k)
                            # dynamic_map = render_flow(v_project.T, render_fn=render_fn)
                            dynamic_map = render_flow(torch.norm(render_pkg["velocity"], dim=-1, keepdim=True), render_fn=render_fn)
                        # dynamic_map = torch.norm(dynamic_map.detach(), dim=0)
                        dynamic_map = dynamic_map[0]
                        # dynamic_map = dynamic_map / dynamic_map.max()
                        dynamic_map = torch.clamp(dynamic_map / dynamic_map.mean() / 4, min=0, max=1)
                        # image_attn_plot = (image_tensor[0] * dynamic_map).permute(1, 2, 0).detach().cpu().numpy()
                        Ll1 = 1.0 * Ll1 + 0.1 * l1_loss_weighted(image_tensor, gt_image_tensor, dynamic_map)

                    if flow_bwd is not None:
                        dt = prev_cam.time - viewpoint_cam.time
                        uv00_pred, _ = project((means3D_deform + velocity * dt).T, prev_cam.w2c.cuda(), prev_cam.k.cuda())
                        flow_bwd_project_pred = (uv00_pred - uv0_pred).T
                        if opt.flow_render:
                            # flow_bwd_project_pred = torch.clamp(flow_bwd_project_pred, min=-W // 2, max=W // 2)
                            flow_bwd_render_pred = render_flow(flow_bwd_project_pred, render_fn=render_fn)
                            flow_bwd_pred_loss = flow_loss_fn(flow_bwd_render_pred.permute(1, 2, 0)[~occ_bwd], flow_bwd[~occ_bwd])
                        else:
                            flow_bwd_pred_loss = flow_loss_fn(flow_bwd_project_pred, flow_bwd_uv0)
                        loss += lambda_flow * 0.1 * flow_bwd_pred_loss

            # print(f"Pixels: {seg_map.sum().int()}, Gaussians: {gaussians.get_xyz.shape[0]}, Overhead: {time.time() - tic:.4}s")

        loss += Ll1

        if torch.isnan(loss):
            # breakpoint()
            raise RuntimeError("Loss is NaN")
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):

                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/{stage}_" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None, load_iteration=args.load_iteration)
    timer.start()
    if checkpoint is None or os.path.basename(checkpoint).split("_")[0] == "coarse":
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
    dataset.read_flow_fwd = opt.use_flow
    dataset.read_flow_bwd = opt.use_flow and opt.flow_bwd
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config["name"] == "test" and stage == "fine":
                    scene.save_best(iteration, psnr_test.item())

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(6666)
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    hp = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()[0])
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7000 + gpu_id)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 500 for i in range(120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3000, 7_000, 8000, 9000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 6000, 10000, 20000, 30000, 60000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")

    torch.cuda.empty_cache()
    if args.model_path:
        os.system(f"python render.py --model_path {args.model_path} --configs {args.configs} --skip_train")
        os.system(f"python metrics.py --model_path {args.model_path}")
