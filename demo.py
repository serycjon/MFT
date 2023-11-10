# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import cv2
import torch
from tqdm import tqdm
import einops

from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import MFT.utils.vis_utils as vu
import MFT.utils.io as io_utils
from MFT.utils.misc import ensure_numpy


logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--video', help='path to a source video (or a directory with images)', type=Path,
                        default=Path('demo_in/ugsJtsO9w1A-00.00.24.457-00.00.29.462_HD.mp4'))
    parser.add_argument('--edit', help='path to a RGBA png with a first-frame edit', type=Path,
                        default=Path('demo_in/edit.png'))
    parser.add_argument('--config', help='MFT config file', type=Path, default=Path('configs/MFT_cfg.py'))
    parser.add_argument('--out', help='output directory', type=Path, default=Path('demo_out/'))
    parser.add_argument('--grid_spacing', help='distance between visualized query points', type=int, default=30)

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args

def run(args):
    config = load_config(args.config)
    logger.info("Loading tracker")
    tracker = config.tracker_class(config)
    logger.info("Tracker loaded")

    initialized = False
    queries = None

    results = []

    logger.info("Starting tracking")
    for frame in tqdm(io_utils.get_video_frames(args.video), total=io_utils.get_video_length(args.video)):
        if not initialized:
            meta = tracker.init(frame)
            initialized = True
            queries = get_queries(frame.shape[:2], args.grid_spacing)
        else:
            meta = tracker.track(frame)

        coords, occlusions = convert_to_point_tracking(meta.result, queries)
        result = meta.result
        result.cpu()
        results.append((result, coords, occlusions))

    edit = None
    if args.edit.exists():
        edit = cv2.imread(str(args.edit), cv2.IMREAD_UNCHANGED)

    logger.info("Drawing the results")
    video_name = args.video.stem
    point_writer = vu.VideoWriter(args.out / f'{video_name}_points.mp4', fps=15, images_export=False)
    if edit is not None:
        edit_writer = vu.VideoWriter(args.out / f'{video_name}_edit.mp4', fps=15, images_export=False)
    for frame_i, frame in enumerate(tqdm(io_utils.get_video_frames(args.video),
                                         total=io_utils.get_video_length(args.video))):
        result, coords, occlusions = results[frame_i]

        dot_vis = draw_dots(frame, coords, occlusions)
        if edit is not None:
            edit_vis = draw_edit(frame, result, edit)
        if False:
            cv2.imshow("cv: dot vis", dot_vis)
            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    sys.exit(1)
                elif c == ord(' '):
                    break

        point_writer.write(dot_vis)
        if edit is not None:
            edit_writer.write(edit_vis)
    point_writer.close()
    if edit is not None:
        edit_writer.close()
    return 0


def get_queries(frame_shape, spacing):
    H, W = frame_shape
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    xs, ys = np.meshgrid(xs, ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    queries = np.vstack((flat_xs, flat_ys)).T
    return torch.from_numpy(queries).float().cuda()

def draw_dots(frame, coords, occlusions):
    canvas = frame.copy()
    N = coords.shape[0]

    for i in range(N):
        occl = occlusions[i] > 0.5
        if not occl:
            thickness = 1 if occl else -1
            vu.circle(canvas, coords[i, :], radius=3, color=vu.RED, thickness=thickness)

    return canvas

def draw_edit(frame, result, edit):
    occlusion_in_template = result.occlusion
    template_visible_mask = einops.rearrange(occlusion_in_template, '1 H W -> H W') < 0.5
    template_visible_mask = template_visible_mask.cpu()
    edit_mask = torch.from_numpy(edit[:, :, 3] > 0)
    template_visible_mask = torch.logical_and(template_visible_mask, edit_mask)

    edit_alpha = einops.rearrange(edit[:, :, 3], 'H W -> H W 1').astype(np.float32) / 255.0
    premult = edit[:, :, :3].astype(np.float32) * edit_alpha
    color_transfer = ensure_numpy(result.warp_forward(premult, mask=template_visible_mask))
    color_transfer = np.clip(color_transfer, 0, 255).astype(np.uint8)
    alpha_transfer = ensure_numpy(result.warp_forward(
        einops.rearrange(edit[:, :, 3], 'H W -> H W 1'),
        mask=template_visible_mask
    ))
    vis = vu.blend_with_alpha_premult(color_transfer, vu.to_gray_3ch(frame), alpha_transfer)
    return vis

from ipdb import iex
@iex
def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    results = main()
