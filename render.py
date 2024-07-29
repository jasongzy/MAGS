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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import imageio
from utils.flow_utils import get_neighbor_cams, project, render_flow, flow_to_image

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flow")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(flow_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    means3D_list = []
    # gaussians.prune_points(gaussians.get_opacity.squeeze(-1) < 0.01)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", dynamic_ncols=True)):
        if idx == 0:
            time1 = time()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        means3D_deform = render_pkg["means3D_deform"]
        if True and name in ("train", "test"):
            _, prev_view = get_neighbor_cams(views, idx, current_cam=view, use_next=False)
            if prev_view is not None:
                uv1, _ = project(means3D_deform.T, view.w2c.cuda(), view.k.cuda())
                uv0, _ = project(means3D_list[idx - 1].T, prev_view.w2c.cuda(), prev_view.k.cuda())
                flow_project = (uv1 - uv0).T
                flow_render = render_flow(
                    flow_project,
                    render_fn=lambda flow2render: render(
                        viewpoint_camera=prev_view,
                        pc=gaussians,
                        pipe=pipeline,
                        bg_color=background,
                        override_color=flow2render,
                    ),
                )
                flow_vis = flow_to_image(flow_render.permute(1, 2, 0).cpu().numpy())
                imageio.imwrite(os.path.join(flow_path, '{0:05d}'.format(idx -1) + ".png"), flow_vis)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        means3D_list.append(means3D_deform)
        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-2, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.predict_flow = False
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)