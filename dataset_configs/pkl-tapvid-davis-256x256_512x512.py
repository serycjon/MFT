from pathlib import Path
from MFT.config import Config
from MFT.environment import env_settings


def get_config():
    conf = Config()
    conf.pickles = [env_settings().tapvid_davis_dir]
    conf.scaling = '256x256_512x512'
    conf.name = Path(__file__).stem
    return conf
