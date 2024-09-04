from mft.MFT import MFT
from pathlib import Path
from mft.MFT.config import Config, load_config
import numpy as np
from mft.MFT.MFT import MFT  # Import the MFT class

tracker_class = MFT  # Reference the class directly


import logging
logger = logging.getLogger(__name__)


def get_config():
    conf = Config()

    conf.tracker_class = MFT
    conf.flow_config = load_config('external/manual_repos/MFT/mft/configs/flow/RAFTou_kubric_huber_split_nonoccl.py')
    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.02

    conf.name = Path(__file__).stem
    return conf
