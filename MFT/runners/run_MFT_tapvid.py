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
import shutil
import torch

from MFT.utils.various import with_debugger, SparseExceptionLogger
from MFT.config import load_config
import MFT.utils.vis_utils as vu
import MFT.utils.io as io_utils
from MFT.utils.telegram_notification import send_notification
from MFT.utils.repro import code_export
from MFT.evaluation import tapvid_eval_stuff as tves
from MFT.point_tracking import convert_to_point_tracking

import logging
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='/datagrid/public_datasets/TAP-Vid', help='dataset config', type=Path)
    parser.add_argument('trackers', default='MFT/configs/MFT_cfg.py', help='path to tracker configs, all must share the same flow_config', type=Path,
                        nargs='+')
    parser.add_argument('--export', default='./export', help='result export directory', type=Path)
    parser.add_argument('--cache', default='./cache', help='flow cache directory', type=Path)
    parser.add_argument('--gpu', help='cuda device')
    parser.add_argument('-c', '--cont', help='skip already computed sequences', action='store_true')
    parser.add_argument('--debug', help='track with tracker debug info', action='store_true')
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--mode', help='TAP-Vid evaluation query modes', choices=['first', 'strided', 'both'],
                        default='both')
    parser.add_argument('--write_flow', help='write flowou for the frame 0 template', action='store_true')
    parser.add_argument('-rcl', '--ram_cache_limit', help='RAM cache limit in GB', type=int, default=30)
    parser.add_argument('-gcl', '--gpu_cache_limit', help='GPU cache limit in GB', type=int, default=5)
    parser.add_argument('--seq', help='sequence subset', nargs='+')
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

    config_name = args.trackers[0].stem

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.export is not None:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{config_name}__{stamp}.log'
        file_handler = logging.FileHandler(log_path)
        log_handlers.append(file_handler)

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


# @profile
def run(args):
    configs = [load_config(path) for path in args.trackers]
    validate_configs(configs)

    # we will load the tracker from the first config, but we have just
    # validated that all the configs share tracker class and flow so
    # we will just monkeypatch the appropriate config into the tracker
    # at runtime. Let's hope nothing breaks :D
    config = configs[0]
    tracker_cls = config.tracker_class
    tracker = tracker_cls(config)

    dataset_conf = load_config(args.dataset)

    for config in configs:
        export_dir = args.export / config.name

        code_path = export_dir / 'code'
        code_path.mkdir(parents=True, exist_ok=True)
        code_export(code_path)
        result_dir = export_dir / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'both':
        query_modes = ['first', 'strided']
    else:
        query_modes = [args.mode]

    for pickle_path in tqdm.tqdm(dataset_conf.pickles, desc='pkl shards', position=0, leave=None, ascii=True):
        dataset = tves.create_tapvid_dataset(pickle_path, query_modes, dataset_conf.scaling)
        for seq in tqdm.tqdm(dataset, desc='sequences', position=1, leave=None, ascii=True):
            cache_dirs = []
            orig_sequence_name = seq['video_name']
            if args.seq is not None and orig_sequence_name not in args.seq:
                continue
            video = seq["data"][query_modes[0]]["video"]  # all query_modes have the same ["video"]
            video = einops.rearrange(video, '1 N_frames H W C -> N_frames H W C', C=3)
            video = video[:, :, :, ::-1].copy()  # convert from RGB to GBR and make contiguous
            assert video.dtype == np.uint8

            flow_name = configs[0].flow_config.name
            assert flow_name
            flow_cache_dir = args.cache / dataset_conf.name / flow_name / orig_sequence_name
            try:
                shutil.rmtree(flow_cache_dir)
            except Exception:
                pass
            flow_cache_dir.mkdir(parents=True, exist_ok=True)
            ram_flow_cache = io_utils.FlowCache(flow_cache_dir,
                                                max_RAM_MB=args.ram_cache_limit * 1e3,
                                                max_GPU_RAM_MB=args.gpu_cache_limit * 1e3)
            cache_dirs.append(flow_cache_dir)

            for query_mode in tqdm.tqdm(query_modes, desc='query mode', position=2, leave=None, ascii=True):
                gt_data = seq["data"][query_mode]
                query_points = einops.rearrange(gt_data['query_points'],
                                                '1 N_queries txy -> N_queries txy').astype(np.int64)
                start_frames = np.unique(query_points[:, 0])
                if query_mode == 'first' and args.write_flow and 0 not in start_frames:
                    raise Exception("Trying to export flowous from first frame, but 0 is not in 'start_frames'" +
                                    f"in {orig_sequence_name} in {pickle_path.stem}")
                N_queries = query_points.shape[0]
                N_frames = video.shape[0]
                xy = 2
                for tracker_config in tqdm.tqdm(configs, desc='trackers', position=3, leave=None, ascii=True):
                    tracker.C = tracker_config
                    pred_occluded = np.zeros((N_queries, N_frames))
                    pred_tracks = np.zeros((N_queries, N_frames, xy))

                    export_dir = args.export / tracker_config.name
                    result_dir = export_dir / 'results'
                    seq_querymode_tracker_result_path = result_dir / f'{orig_sequence_name}-{query_mode}.pklz'
                    if args.cont and seq_querymode_tracker_result_path.exists():
                        print(
                            f'skipping {orig_sequence_name}-{query_mode} for {tracker_config.name} - already computed')
                        continue
                    # accumulate the tracker results for all the query
                    # points (and both directions if "strided" evaluation mode)
                    for start_frame in tqdm.tqdm(start_frames, desc='query frame', position=4, leave=None, ascii=True):
                        current_mask = query_points[:, 0] == start_frame
                        current_queries = query_points[current_mask, 1:]
                        current_queries = current_queries[:, ::-1].copy()  # convert to xy order, make contiguous
                        torch_current_queries = torch.from_numpy(current_queries).to('cuda')

                        directions = ['forward']
                        if query_mode == 'strided':
                            directions.append('backward')
                        for direction in directions:
                            sequence_name = f'{orig_sequence_name}--{start_frame}--{direction}'
                            if export_dir is not None:
                                seq_result_dir = result_dir / sequence_name
                                seq_result_dir.mkdir(parents=True, exist_ok=True)
                                meta_path = seq_result_dir / 'meta.pklz'
                            try:
                                metas = track_sequence(tracker, video, start_frame, direction=direction,
                                                       debug=args.debug, flow_cache=ram_flow_cache)
                            except KeyboardInterrupt:
                                raise
                            except Exception:
                                logger.exception(f'error in sequence {sequence_name}')
                                raise

                            frame_i_gen = range(start_frame, N_frames)
                            if direction == 'backward':
                                frame_i_gen = range(start_frame, 0 - 1, -1)
                            for frame_i in frame_i_gen:
                                meta = metas[frame_i]
                                current_coords, current_occlusions =  convert_to_point_tracking(meta.result, torch_current_queries)
                                pred_tracks[current_mask, frame_i, :] = current_coords
                                pred_occluded[current_mask, frame_i] = current_occlusions

                            # export {{{
                            if args.export is not None:
                                if any([hasattr(meta, 'vis') for meta in metas.values()]):
                                    vis_path = Path(export_dir) / 'vis'
                                    vis_path.mkdir(parents=True, exist_ok=True)
                                    writer = vu.VideoWriter(vis_path / f'{sequence_name}.mp4')
                                    for frame_i in sorted(list(metas.keys())):
                                        meta = metas[frame_i]
                                        vis = getattr(meta, 'vis', None)
                                        if vis is not None:
                                            writer.write(vis)
                                    writer.close()

                                if start_frame == 0 and query_mode == 'first' and args.write_flow:
                                    flowou_dir = Path(export_dir) / 'flowous' / orig_sequence_name
                                    flowou_dir.mkdir(parents=True, exist_ok=True)
                                    for frame_i in frame_i_gen:
                                        meta = metas[frame_i]
                                        result = meta.result
                                        flowou_path = flowou_dir / f'0--{frame_i}.flowouX16.pkl'
                                        result.write(flowou_path)

                                if False:
                                    metas = {k: prune_meta(v, ['vis', 'result']) for k, v in metas.items()}
                                    with open(meta_path, 'wb') as fout:
                                        pickle.dump(metas, fout)
                            # }}}
                    # sequence finished for one tracker and one query modeone mode finished, save the tracklets
                    H, W = video.shape[1], video.shape[2]
                    pred_occluded = einops.rearrange(pred_occluded, 'N_queries N_frames -> 1 N_queries N_frames')
                    pred_tracks = einops.rearrange(pred_tracks,
                                                   'N_queries N_frames xy -> 1 N_queries N_frames xy', xy=2)
                    scale = einops.rearrange(np.array([256.0 / W, 256.0 / H]), 'xy -> 1 1 1 xy')
                    pred_tracks *= scale
                    assert pred_tracks.shape[0] == 1
                    assert pred_tracks.shape[3] == 2
                    assert len(pred_tracks.shape) == 4
                    tracklet_outputs = {'tracks': pred_tracks,
                                        'occluded': pred_occluded}
                    with open(seq_querymode_tracker_result_path, 'wb') as fout:
                        pickle.dump(tracklet_outputs, fout)

                # sequence finished for all trackers in one query_mode
            # sequence finished for all trackers, all query modes, let's clean up
            for cache_dir in cache_dirs:
                shutil.rmtree(cache_dir)
            ram_flow_cache.clear()

    if args.export is not None:
        send_notification(f"MFT TAP-Vid run finished (`{' '.join(sys.argv)}`)", use_markdown=True)
    return 0


# @profile
def track_sequence(tracker, video, start_frame, direction='forward', debug=False, flow_cache=None):
    assert direction in ['forward', 'backward']
    all_metas = {}
    sparse_logger = SparseExceptionLogger(logger)

    N_frames = video.shape[0]
    initialized = False
    frame_i_gen = range(start_frame, N_frames)
    time_direction = +1
    if direction == 'backward':
        frame_i_gen = range(start_frame, 0 - 1, -1)
        time_direction = -1

    for frame_i in frame_i_gen:
        frame = video[frame_i, :, :, :]

        if not initialized:
            initialized = True
            meta = tracker.init(frame, start_frame_i=start_frame, time_direction=time_direction,
                                flow_cache=flow_cache)
        else:
            try:
                meta = tracker.track(frame, debug=debug)
            except StopIteration:
                break
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                sparse_logger("Tracker exception", ex)
                raise
        meta.frame_i = frame_i
        meta.backward = (direction == 'backward')

        all_metas[frame_i] = meta
    return all_metas


def prune_meta(meta, to_prune=None):
    if to_prune is None:
        to_prune = []
    meta_keys = list(meta.__dict__.keys())
    for key in meta_keys:
        if key in to_prune:
            delattr(meta, key)

    return meta


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
