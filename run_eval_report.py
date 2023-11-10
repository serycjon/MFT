# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import logging
import re
import sys

import pandas as pd
from tabulate import tabulate

from MFT.runners.eval_MFT_tapvid import run as run_evaluation
from MFT.runners.run_MFT_tapvid import parse_arguments
from MFT.runners.run_MFT_tapvid import run as run_tracker

pd.set_option('display.precision', 1)
logger = logging.getLogger(__name__)


def method_rename(config_name):
    # config_name = re.sub(r"^MPT_multiflow_occl_sigmasq_occlinvalid", "MFT", config_name)
    config_name = re.sub(r"_cfg$", "", config_name)
    return config_name


def run(args):
    try:
        run_tracker(args)
    except Exception:
        logger.exception("Tracking failed")
        # raise
    run_evaluation(args)
    report(args)

    return 0


def report(args):
    print('FIRST:')
    report_first(args)
    print('\n\nSTRIDED:')
    report_strided(args)


def report_first(args):
    report_aux(args, 'tapvid-eval.pklz')

def report_strided(args):
    report_aux(args, 'tapvid-eval-strided.pklz')

def report_aux(args, pickle_name):
    all_methods = []
    res_dir = args.export
    paths = sorted(list(res_dir.glob(f'*/eval/{pickle_name}')))
    for path in paths:
        method_name = path.parent.parent.stem
        method_df = pd.read_pickle(path)
        
        # print(method_df)
        try:
            method_results = method_df[['average_prec', 'average_pts_within_thresh',
                                        'pts_within_1', 'pts_within_2', 'pts_within_4',
                                        'pts_within_8', 'pts_within_16', 'occlusion_accuracy',
                                        'average_jaccard']].mean() * 100
        except KeyError:
            continue
        method_results['method'] = method_name
        # method_results['resolution'] = resolution
        method_results = method_results.to_frame().T
       
        all_methods.append(method_results)
    results = pd.concat(all_methods)
    # assuming TAP-Vid DAVIS
    if 'strided' in pickle_name:
        results = results.append({'average_pts_within_thresh': 53.1, 'occlusion_accuracy': 82.3, 'average_jaccard': 38.4,
                                  'method': 'TAP-Net', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 59.4, 'occlusion_accuracy': 82.1, 'average_jaccard': 42.0,
                                  'method': 'PIPs', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 67.5, 'occlusion_accuracy': 85.3, 'average_jaccard': 51.7,
                                  'method': 'OmniMotion', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 72.3, 'occlusion_accuracy': 87.6, 'average_jaccard': 61.3,
                                  'method': 'TAPIR', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 79.1, 'occlusion_accuracy': 88.7, 'average_jaccard': 64.8,
                                  'method': 'CoTracker', 'resolution': '256'}, ignore_index=True)
    else:
        results = results.append({'average_pts_within_thresh': 48.6, 'occlusion_accuracy': 78.8, 'average_jaccard': 33.0,
                                  'method': 'TAP-Net', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 70.0, 'occlusion_accuracy': 86.5, 'average_jaccard': 56.2,
                                  'method': 'TAPIR', 'resolution': '256'}, ignore_index=True)
        results = results.append({'average_pts_within_thresh': 75.4, 'occlusion_accuracy': 89.3, 'average_jaccard': 60.6,
                                  'method': 'CoTracker', 'resolution': '256'}, ignore_index=True)
    results['cfg'] = results['method']
    results['method'] = results['method'].apply(method_rename)
    first_column = results.pop('method')
    results.insert(0, 'method', first_column)

    results = results.rename(columns={'average_pts_within_thresh': '< thrs',
                                      'occlusion_accuracy': 'OA',
                                      'average_jaccard': 'AJ',
                                      'pts_within_1': '< 1',
                                      'pts_within_2': '< 2',
                                      'pts_within_4': '< 4',
                                      'pts_within_8': '< 8',
                                      'pts_within_16': '< 16'})
    results = results[['method', 'AJ', '< thrs', 'OA', '< 1', '< 2', '< 4', '< 8', '< 16']]

    print(tabulate(results, headers="keys", tablefmt="orgtbl", floatfmt=".2f"))


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
