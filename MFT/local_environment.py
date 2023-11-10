import os

def local_env_settings():
    settings = EnvironmentSettings()
    return settings


class EnvironmentSettings:
    def __init__(self):
        self.tapvid_davis_dir = '/mnt/datasets/TAP-Vid/tapvid_davis/tapvid_davis.pkl'
