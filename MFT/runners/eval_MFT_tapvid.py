# -*- origami-fold-style: triple-braces -*-
import sys
import os
import argparse
import tqdm
import numpy as np
import pickle
from pathlib import Path
import socket
import datetime
import einops
from collections import defaultdict
import pandas as pd

from MFT.utils.various import with_debugger
from MFT.config import load_config
from MFT.evaluation import tapvid_eval_stuff as tves

import logging
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='dataset config', type=Path)
    parser.add_argument('trackers', help='path to tracker configs, all must share the same flow_config', type=Path,
                        nargs='+')
    parser.add_argument('--export', help='result export directory', type=Path, required=True)
    parser.add_argument('--cache', help='flow cache directory', type=Path, required=True)
    parser.add_argument('--gpu', help='cuda device')
    parser.add_argument('-c', '--cont', help='skip already computed sequences', action='store_true')
    parser.add_argument('--debug', help='track with tracker debug info', action='store_true')
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--mode', help='TAP-Vid evaluation query modes', choices=['first', 'strided', 'both'],
                        default='both')
    return parser


def parse_arguments():
    parser = get_parser()

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    stdout_lvl = logging.DEBUG if args.verbose else logging.INFO
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(stdout_lvl)
    log_handlers = [stdout_handler]

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    log_format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=log_handlers)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("ltr.admin.loading").setLevel(logging.ERROR)

    hostname = socket.gethostname()
    cmdline = str(' '.join(sys.argv))
    logger.info(f"cmd: {cmdline}")
    logger.info(f"start: {stamp}")
    logger.info(f"host: {hostname}")

    return args


def run(args):
    configs = [load_config(path) for path in args.trackers]

    dataset_conf = load_config(args.dataset)

    if args.mode == 'both':
        query_modes = ['first', 'strided']
    else:
        query_modes = [args.mode]

    all_metrics = {'strided': defaultdict(list),
                   'first': defaultdict(list)}
    for pickle_path in tqdm.tqdm(dataset_conf.pickles, desc='pkl shards', position=0, leave=None, ascii=True):
        dataset = tves.create_tapvid_dataset(pickle_path, query_modes, dataset_conf.scaling, fake_video=True)
        for seq in tqdm.tqdm(dataset, desc='sequences', position=1, leave=None, ascii=True):
            orig_sequence_name = seq['video_name']
            video = seq["data"][query_modes[0]]["video"]  # all query_modes have the same ["video"]
            video = einops.rearrange(video, '1 N_frames H W C -> N_frames H W C', C=3)
            assert video.dtype == np.uint8

            for query_mode in tqdm.tqdm(query_modes, desc='query mode', position=2, leave=None, ascii=True):
                gt_data = seq["data"][query_mode]
                query_points = einops.rearrange(gt_data['query_points'],
                                                '1 N_queries txy -> N_queries txy').astype(np.int64)
                gt_tracks = gt_data['target_points']  # (1, N_queries, N_frames, 2) in dataset-config-specific scale
                H, W = video.shape[1], video.shape[2]
                scale = einops.rearrange(np.array([256.0 / W, 256.0 / H]), 'xy -> 1 1 1 xy')
                gt_tracks *= scale
                gt_occluded = gt_data['occluded']  # (1, N_queries, N_frames)
                for tracker_config in tqdm.tqdm(configs, desc='trackers', position=3, leave=None, ascii=True):
                    export_dir = args.export / tracker_config.name
                    result_dir = export_dir / 'results'
                    seq_querymode_tracker_result_path = result_dir / f'{orig_sequence_name}-{query_mode}.pklz'
                    with open(seq_querymode_tracker_result_path, 'rb') as fin:
                        tracklet_outputs = pickle.load(fin)
                        pred_tracks = tracklet_outputs['tracks']  # (1, N_queries, N_frames), scaled to 256 x 256
                        pred_occluded = tracklet_outputs['occluded']  # (1, N_queries, N_frames, xy)

                    pred_occluded = np.float32(pred_occluded > 0.5)
                    assert pred_tracks.shape[0] == 1
                    assert pred_tracks.shape[3] == 2
                    assert len(pred_tracks.shape) == 4
                    assert gt_occluded.shape == pred_occluded.shape
                    assert gt_tracks.shape == pred_tracks.shape
                    metrics = tves.compute_tapvid_metrics(
                        query_points,
                        gt_occluded,
                        gt_tracks,
                        pred_occluded,
                        pred_tracks,
                        query_mode)
                    # skip singleton dimension:
                    assert all(val.shape == (1, ) for key, val in metrics.items())
                    metrics = {key: val[0] for key, val in metrics.items()}
                    metrics['seq'] = orig_sequence_name
                    all_metrics[query_mode][tracker_config.name].append(metrics)

    for tracker_config in tqdm.tqdm(configs, desc='export', ascii=True):
        tracker_name = tracker_config.name
        export_dir = args.export / tracker_name
        eval_dir = export_dir / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)

        for query_mode in query_modes:
            tracker_metrics = all_metrics[query_mode][tracker_name]

            tracker_metrics = {i: val for i, val in enumerate(tracker_metrics)}
            tracker_metrics = pd.DataFrame.from_dict(tracker_metrics, orient="index")
            out_name = 'tapvid-eval'
            if query_mode == 'strided':
                out_name += '-strided'
            tracker_metrics.to_pickle(eval_dir / f'{out_name}.pklz')
    return 0


def all_same(xs):
    return all(x == xs[0] for x in xs)


def validate_configs(configs):
    # check that all the configs share the same tracker class and optical flow
    assert all_same([c.tracker_class for c in configs])
    assert all_same([c.flow_config for c in configs])


@with_debugger
def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
