#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
import cv2
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.utils.misc import to_cpu, save_conf_maps, save_dynamic_masks, save_depth_maps, save_rgb_imgs, save_focals, save_intrinsics, save_tum_poses
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format
from dust3r.utils.flow_vis import show_flow, flow_to_image

import matplotlib.pyplot as pl
# from third_party.raft import load_RAFT
from datasets_preprocess.sintel_get_dynamics import compute_optical_flow
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05, show_cam=True,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, save_name=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None: save_name='scene'
    outfile = os.path.join(outdir, save_name+'.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    # scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)

def get_dynamic_mask_from_pairviewer(scene, flow_net=None, both_directions=False):
    """
    get the dynamic mask from the pairviewer
    """
    if flow_net is None:
        # flow_net = load_RAFT(model_path="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth").to('cuda').eval() # sea-raft
        flow_net = load_RAFT(model_path="third_party/RAFT/models/raft-things.pth").to('cuda').eval()

    imgs = scene.imgs
    img1 = torch.from_numpy(imgs[0]*255).permute(2,0,1)[None]                       # (B, 3, H, W)
    img2 = torch.from_numpy(imgs[1]*255).permute(2,0,1)[None]
    with torch.no_grad():
        forward_flow = flow_net(img1.cuda(), img2.cuda(), iters=20, test_mode=True)[1]  # (B, 2, H, W)
        if both_directions:
            backward_flow = flow_net(img2.cuda(), img1.cuda(), iters=20, test_mode=True)[1]
            
    B, _, H, W = forward_flow.shape

    depth_map1 = scene.get_depthmaps()[0] # (H, W)
    depth_map2 = scene.get_depthmaps()[1]

    im_poses = scene.get_im_poses()
    cam1 = im_poses[0]                  # (4, 4)   cam2world
    cam2 = im_poses[1]
    extrinsics1 = torch.linalg.inv(cam1) # (4, 4)   world2cam
    extrinsics2 = torch.linalg.inv(cam2)

    intrinsics = scene.get_intrinsics()
    intrinsics_1 = intrinsics[0]        # (3, 3)
    intrinsics_2 = intrinsics[1]

    ego_flow_1_2 = compute_optical_flow(depth_map1, depth_map2, extrinsics1, extrinsics2, intrinsics_1, intrinsics_2) # (H*W, 2)
    ego_flow_1_2 = ego_flow_1_2.reshape(H, W, 2).transpose(2, 0, 1) # (2, H, W)

    error_map = np.linalg.norm(ego_flow_1_2 - forward_flow[0].cpu().numpy(), axis=0) # (H, W)

    error_map_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    error_map_normalized_int = (error_map_normalized * 255).astype(np.uint8)
    if both_directions:
        ego_flow_2_1 = compute_optical_flow(depth_map2, depth_map1, extrinsics2, extrinsics1, intrinsics_2, intrinsics_1)
        ego_flow_2_1 = ego_flow_2_1.reshape(H, W, 2).transpose(2, 0, 1)
        error_map_2 = np.linalg.norm(ego_flow_2_1 - backward_flow[0].cpu().numpy(), axis=0)
        error_map_2_normalized = (error_map_2 - error_map_2.min()) / (error_map_2.max() - error_map_2.min())
        error_map_2_normalized = (error_map_2_normalized * 255).astype(np.uint8)
        cv2.imwrite('./demo_tmp/dynamic_mask_bw.png', cv2.applyColorMap(error_map_2_normalized, cv2.COLORMAP_JET))
        np.save('./demo_tmp/dynamic_mask_bw.npy', error_map_2)

        backward_flow = backward_flow[0].cpu().numpy().transpose(1, 2, 0)
        np.save('./demo_tmp/backward_flow.npy', backward_flow)
        flow_img = flow_to_image(backward_flow)
        cv2.imwrite('./demo_tmp/backward_flow.png', flow_img)

    cv2.imwrite('./demo_tmp/dynamic_mask.png', cv2.applyColorMap(error_map_normalized_int, cv2.COLORMAP_JET))
    error_map_normalized_bin = (error_map_normalized > 0.35).astype(np.uint8)
    # save the binary mask
    cv2.imwrite('./demo_tmp/dynamic_mask_binary.png', error_map_normalized_bin*255)
    # save the original one as npy file
    np.save('./demo_tmp/dynamic_mask.npy', error_map)

    # also save the flow
    forward_flow = forward_flow[0].cpu().numpy().transpose(1, 2, 0)
    np.save('./demo_tmp/forward_flow.npy', forward_flow)
    # save flow as image
    flow_img = flow_to_image(forward_flow)
    cv2.imwrite('./demo_tmp/forward_flow.png', flow_img)

    return error_map


def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_self_mask,
                            init_data=None, skip_save=False):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=use_self_mask,
                               partial_init=init_data is not None, num_total_iter=niter, empty_cache= len(filelist) > 72)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    # use the init data here
    if init_data is not None:
        init_poses, init_intrinsics, init_depth_maps = init_data
        msk = np.zeros(len(filelist), dtype=bool)
        msk[:len(init_poses)] = True
        scene.preset_pose(init_poses, msk)
        scene.preset_intrinsics(init_intrinsics, msk)
        scene.preset_depthmap(init_depth_maps, msk)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    if not skip_save:

        if seq_name != "NULL" and seq_name.startswith("sintel"):
            pred_traj = scene.get_tum_poses()
            model_id = new_model_weights.split('/')[1]
            seq_name = seq_name.split('-')[-1]
            scene.save_tum_poses(f'./tmp/{seq_name}_pred_traj_{model_id}.txt')
            gt_traj = load_traj(gt_traj_file=f'data/sintel/training/camdata_left/{seq_name}', stride=5)
            eval_metrics(pred_traj, gt_traj, seq=seq_name, filename=f"./tmp/{seq_name}_tmp_metric_{model_id}.log")
            plot_trajectory(pred_traj, gt_traj, title=seq_name, filename=f"./tmp/{seq_name}_{model_id}.png")
        elif seq_name != "NULL":
            save_folder = f'./tmp/{seq_name}'
            os.makedirs(save_folder, exist_ok=True)
            poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
            K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
            depth_maps = scene.save_depth_maps(save_folder)
            dynamic_masks = scene.save_dynamic_masks(save_folder)
            conf = scene.save_conf_maps(save_folder)
            init_conf = scene.save_init_conf_maps(save_folder)
            rgbs = scene.save_rgb_imgs(save_folder)
            enlarge_seg_masks(save_folder, kernel_size=3 if use_self_mask else 5) 

        outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                        clean_depth, transparent_cams, cam_size, show_cam)

        # also return rgb, depth and confidence imgs
        # depth is normalized with the max value for all images
        # we apply the jet colormap on the confidence maps
        rgbimg = scene.imgs
        depths = to_numpy(scene.get_depthmaps())
        confs = to_numpy([c for c in scene.im_conf])
        init_confs = to_numpy([c for c in scene.init_conf_maps])
        cmap = pl.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        depths = [cmap(d/depths_max) for d in depths]
        confs_max = max([d.max() for d in confs])
        confs = [cmap(d/confs_max) for d in confs]
        init_confs_max = max([d.max() for d in init_confs])
        init_confs = [cmap(d/init_confs_max) for d in init_confs]

        imgs = []
        for i in range(len(rgbimg)):
            imgs.append(rgbimg[i])
            imgs.append(rgb(depths[i]))
            imgs.append(rgb(confs[i]))
            imgs.append(rgb(init_confs[i]))

        # if two images, and the shape is same
        if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
            error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True)
            # imgs.append(rgb(error_map))
            # apply threshold on the error map
            normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
            error_map_max = normalized_error_map.max()
            error_map = cmap(normalized_error_map/error_map_max)
            imgs.append(rgb(error_map))
            binary_error_map = (normalized_error_map > 0.35).astype(np.uint8)
            imgs.append(rgb(binary_error_map*255))
    else:
        outfile = None
        imgs = None
    return scene, outfile, imgs


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin" or scenegraph_type == "swin2stride" or scenegraph_type == "swinstride":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,5),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value='NULL', info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                use_self_mask = gradio.Checkbox(value=False, label="Use Self Mask")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_self_mask,],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
    demo.launch(share=args.share, server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = get_args_parser()
    # add "share" argument
    parser.add_argument("--share", action='store_true', default=False, help="share the demo")
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
