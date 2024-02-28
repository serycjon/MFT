import subprocess
from MFT import repo_path
import logging
logger = logging.getLogger(__name__)


def git_dirty_p(path=None):
    out = subprocess.check_output(["git", "status",
                                   # "-uno",  # ignore untracked
                                   "--porcelain"], cwd=path)
    return out.strip() != b""


def git_diff(path=None):
    out = subprocess.check_output(["git", "diff", "--no-color"], cwd=path)
    return out.strip()


def git_commit(path=None):
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path)
    return out.strip().decode()


def code_export(path):
    env_path = str(repo_path)  # the main repo directory
    cmd = ["rsync", "-avm",
           # "--dry-run"
           ]

    excluded = ['.git', '__pycache__', 'pytracking_old', '.ipynb_checkpoints',
                'logs', 'export', 'demo-in', 'demo-out', 'runs', 'train_files_lists',
                'cache',
                # RAFT stuff
                'checkpoints', 'flowou_evals', 'traintxt', 'demo-frames',
                'experiments']
    for x in excluded:
        cmd.append(f"--exclude='{x}'")
    path.mkdir(parents=True, exist_ok=True)
    cmd += ["--include='*/'", "--include='*.py'", "--exclude='*'", "./",
            str(path.expanduser().resolve())]
    subprocess.check_call(' '.join(cmd), cwd=env_path, shell=True,
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    logger.info(f'MFT repo backed up at {path}')

