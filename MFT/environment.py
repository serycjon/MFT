import importlib
import os

from MFT import code_path


class EnvSettings:
    def __init__(self):
        self.otb_path = ''
        self.nfs_path = ''
        self.uav_path = ''
        self.tpl_path = ''
        self.vot_path = ''
        self.got10k_path = ''
        self.lasot_path = ''
        self.trackingnet_path = ''
        self.mobiface_path = ''
        self.vot18_path = ''
        self.vot16_path = ''


def create_default_local_file():
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = code_path / 'local_environment.py'
    with open(path, 'w') as f:
        settings = EnvSettings()

        f.write('from MFT.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')


def env_settings():
    env_module_name = 'MFT.local_environment'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = code_path / 'local_environment.py'

        # Create a default file
        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local_environment.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))
