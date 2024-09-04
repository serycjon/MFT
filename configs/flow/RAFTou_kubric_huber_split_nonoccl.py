from pathlib import Path
from mft.MFT.config import Config
from mft.MFT.raft import RAFTWrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)


def get_config():
    conf = Config()

    conf.of_class = RAFTWrapper
    conf_name = Path(__file__).stem

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty',
        'small': False,
        'mixed_precision': False,
    }
    conf.raft_params = AttrDict(**raft_kwargs)
    # original model location:
    conf.model = 'external/manual_repos/MFT/mft/checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth'

    conf.flow_iters = 12

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
