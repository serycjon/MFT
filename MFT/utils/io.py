import sys
import glob
import time
import datetime
from pathlib import Path
from collections import deque, Counter
import cv2
import re
import gzip
import pickle
import io
import numpy as np
import einops
import os
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import shutil
from MFT.utils.misc import ensure_numpy
import tqdm

import logging
logger = logging.getLogger(__name__)


def get_frames(path):
    paths = glob.glob(f'{path}/*.jpg')
    return sorted([Path(path) for path in paths])


def video_seek_frame(time_string, fps=30):
    parsed_time = time.strptime(time_string, "%H:%M:%S")
    delta = datetime.timedelta(hours=parsed_time.tm_hour, minutes=parsed_time.tm_min, seconds=parsed_time.tm_sec)
    time_seconds = int(delta.total_seconds())
    pos = fps * time_seconds
    return pos


def video_seek_frame_name(query_frame_name, frame_paths):
    frame_names = [path.stem for path in frame_paths]
    regexp = re.compile(r'0*' + query_frame_name)
    for i, name in enumerate(frame_names):
        if re.match(regexp, name):
            return i
    raise ValueError(f"Frame {query_frame_name} not found.")


def frames_from_time(directory, time_string, fps=30):
    frames = get_frames(directory)
    start_index = video_seek_frame(time_string, fps)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


def frames_from_name(directory, start_name):
    frames = get_frames(directory)
    start_index = video_seek_frame_name(start_name, frames)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


class LookaheadIter:

    def __init__(self, it):
        self._iter = iter(it)
        self._ahead = deque()

    def __iter__(self):
        return self

    def __next__(self):
        if self._ahead:
            return self._ahead.popleft()
        else:
            return next(self._iter)

    def lookahead(self):
        for x in self._ahead:
            yield x
        for x in self._iter:
            self._ahead.append(x)
            yield x

    def peek(self, *a):
        return next(iter(self.lookahead()), *a)


def load_maybe_gzipped_pkl(path):
    suffix = Path(path).suffix
    if suffix == '.pklz':
        open_fn = gzip.open
    elif suffix == '.pkl':
        open_fn = open
    else:
        ValueError(f"Unknown pickle file suffix ({suffix}).")

    with open_fn(path, 'rb') as fin:
        data = pickle.load(fin)

    return data


class CPU_Unpickler(pickle.Unpickler):
    """ https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    I have pickled something in meta as a GPU tensor..."""

    def find_class(self, module, name):
        import torch

        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_cpu_pickle(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"No pickle at {path}")
    try:
        exception = gzip.BadGzipFile  # new in python 3.8
    except AttributeError:
        exception = OSError

    try:
        with gzip.open(path, 'rb') as fin:
            unpickler = CPU_Unpickler(fin)
            data = unpickler.load()
    except exception:  # we didn't compress this one...
        with open(path, 'rb') as fin:
            unpickler = CPU_Unpickler(fin)
            data = unpickler.load()
    return data


def read_flow_png(path):
    """Read png-compressed flow

    Args:
        path: png flow file path

    Returns:
        flow: (H, W, 2) float32 numpy array (delta-x, delta-y)
        valid: (H, W) float32 numpy array
    """
    # to specify not to change the image depth (16bit)
    flow = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 32.0
    return flow, valid


def write_flow_png(path, flow, valid=None):
    """Write a compressed png flow

    Args:
        path: write path
        flow: (H, W, 2) xy-flow
        valid: None, or (H, W) array with validity mask
    """
    flow = 32.0 * flow + 2**15  # compress (resolution step 1/32, maximal flow 1024 (same as Sintel width))
    if valid is None:
        valid = np.ones([flow.shape[0], flow.shape[1], 1])
    else:
        valid = einops.rearrange(valid, 'H W -> H W 1', **einops.parse_shape(flow, 'H W _'))
    data = np.concatenate([flow, valid], axis=2).astype(np.uint16)
    cv2.imwrite(str(path), data[:, :, ::-1])


# flow is encoded with sign, so 2**15, occlusion and uncertainty without sign, so 2**16:
FLOWOU_IO_FLOW_MULTIPLIER = 2**5  # max-abs-val = 2**(15-5) = 1024, step = 2**(-5) = 0.03
FLOWOU_IO_OCCLUSION_MULTIPLIER = 2**15  # max-val = 2**(16-15) = 2, step = 2**(-15) = 3e-5
FLOWOU_IO_UNCERTAINTY_MULTIPLIER = 2**9   # max-val = 2**(16-9) = 128, step = 2**(-9) = 0.0019


def write_flowou(path, flow, occlusions, uncertainty):
    """Write a compressed png flow, occlusions and uncertainty

    Args:
        path: write path (must have ".flowou.png", ".flowouX16.pkl", or ".flowouX32.pkl" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) array with uncertainty sigma
    """
    suf = Path(path).suffixes[0]
    if suf == '.flowou':
        write_flowou1_png(path, flow, occlusions, uncertainty)
    elif suf == '.flowouX16':
        write_flowou_X16(path, flow, occlusions, uncertainty)
    elif suf == '.flowouX32':
        write_flowou_X32(path, flow, occlusions, uncertainty)
    elif suf == '.stepan16':
        write_flowou_stepan16(path, flow, occlusions, uncertainty)
    else:
        raise ValueError(f"Incorrect flowou path suffix: {Path(path).suffixes}")


def read_flowou(path):
    """Read png-compressed flow, occlusions and uncertainty

    Args:
        path: ".flowou.png", ".flowouX16.pkl", or ".flowouX32.pkl" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
    suf = Path(path).suffixes[0]
    if suf == '.flowou':
        return read_flowou1_png(path)
    elif suf == '.flowouX16':
        return read_flowou_X16(path)
    elif suf == '.flowouX32':
        return read_flowou_X32(path)
    else:
        raise ValueError(f"Incorrect flowou path suffix: {Path(path).suffixes}")


def write_flowou1_png(path, flow, occlusions, uncertainty):
    """Write a compressed png flow, occlusions and uncertainty

    Args:
        path: write path (must have ".flowou.png" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible), clipped between 0 and 1
        uncertainty: (1, H, W) array with uncertainty sigma, clipped between 0 and 2047
                      (0 = dirac, max observed on Sintel = 215, Q0.999 on sintel ~ 15)
    """
    def encode_central(xs, multiplier=32.0):
        max_val = 2**15 / multiplier
        assert np.all(np.abs(xs) < max_val), "out-of-range values - cannot be written"
        return 2**15 + multiplier * xs

    def encode_positive(xs, multiplier=32.0):
        max_val = 2**16 / multiplier
        assert np.all(xs >= 0), "out-of-range values - cannot be written"
        assert np.all(xs < max_val), "out-of-range values - cannot be written"
        return multiplier * xs

    assert Path(path).suffixes == ['.flowou', '.png']
    path.parent.mkdir(parents=True, exist_ok=True)
    einops.parse_shape(flow, 'H W xy')
    flow = encode_central(einops.rearrange(flow, 'xy H W -> H W xy', xy=2),
                          multiplier=FLOWOU_IO_FLOW_MULTIPLIER)

    occlusions = np.clip(occlusions, 0, 1)
    occlusions = encode_positive(einops.rearrange(occlusions, '1 H W -> H W 1', **einops.parse_shape(flow, 'H W _')),
                                 multiplier=FLOWOU_IO_OCCLUSION_MULTIPLIER)

    uncertainty = np.clip(uncertainty, 0, 127)
    uncertainty = encode_positive(einops.rearrange(uncertainty, '1 H W -> H W 1', **einops.parse_shape(flow, 'H W _')),
                                  multiplier=FLOWOU_IO_UNCERTAINTY_MULTIPLIER)

    data = np.concatenate([flow, occlusions, uncertainty], axis=2).astype(np.uint16)
    cv2.imwrite(str(path), data)


def read_flowou1_png(path):
    """Read png-compressed flow, occlusions and uncertainty

    Args:
        path: ".flowou.png" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
    # to specify not to change the image depth (16bit)
    assert Path(path).suffixes == ['.flowou', '.png']

    def decode_central(xs, multiplier=32.0):
        return (xs.astype(np.float32) - 2**15) / multiplier

    def decode_positive(xs, multiplier=32.0):
        return xs.astype(np.float32) / multiplier

    data = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    data = einops.rearrange(data, 'H W C -> C H W', C=4)
    flow, occlusions, uncertainty = data[:2, :, :], data[2, :, :], data[3, :, :]
    occlusions = einops.rearrange(occlusions, 'H W -> 1 H W')
    uncertainty = einops.rearrange(uncertainty, 'H W -> 1 H W')
    flow = decode_central(flow, multiplier=FLOWOU_IO_FLOW_MULTIPLIER)
    occlusions = decode_positive(occlusions, multiplier=FLOWOU_IO_OCCLUSION_MULTIPLIER)
    uncertainty = decode_positive(uncertainty, multiplier=FLOWOU_IO_UNCERTAINTY_MULTIPLIER)
    return flow, occlusions, uncertainty


def write_flowou2_png(path, flow, occlusions, uncertainty):
    """Write a compressed png flow, occlusions and uncertainty, with a variable min-max range

    Args:
        path: write path (must have ".flowou2.png" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible), clipped between 0 and 1
        uncertainty: (1, H, W) array with uncertainty sigma, clipped between 0 and 2047
                      (0 = dirac, max observed on Sintel = 215, Q0.999 on sintel ~ 15)
    """
    def encode(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)

        uint16_xs = np.uint16(xs_01 * (2**16 - 1))
        return uint16_xs, lb, ub

    assert Path(path).suffixes == ['.flowou2', '.png']
    path.parent.mkdir(parents=True, exist_ok=True)
    einops.parse_shape(flow, 'H W xy')
    flow, flow_min, flow_max = encode(einops.rearrange(flow, 'xy H W -> H W xy', xy=2))

    occlusions, occl_min, occl_max = encode(einops.rearrange(occlusions, '1 H W -> H W 1',
                                                             **einops.parse_shape(flow, 'H W _')))

    uncertainty, unc_min, unc_max = encode(einops.rearrange(uncertainty, '1 H W -> H W 1',
                                                            **einops.parse_shape(flow, 'H W _')))

    data = np.concatenate([flow, occlusions, uncertainty], axis=2)
    pil_data = Image.fromarray(data)
    metadata = PngInfo()
    metadata.add_text("flow_min", str(flow_min))
    metadata.add_text("flow_max", str(flow_max))

    metadata.add_text("occl_min", str(occl_min))
    metadata.add_text("occl_max", str(occl_max))

    metadata.add_text("unc_min", str(unc_min))
    metadata.add_text("unc_max", str(unc_max))
    pil_data.save(str(path), pnginfo=metadata)


def read_flowou2_png(path):
    """Read png-compressed flow, occlusions and uncertainty, with a variable min-max range

    Args:
        path: ".flowou2.png" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
    # to specify not to change the image depth (16bit)
    assert Path(path).suffixes == ['.flowou2', '.png']

    def decode(xs, lb, ub):
        xs_01 = np.float32(xs) / (2**16 - 1)
        return lb + xs_01 * (ub - lb)

    pil_data = Image.open(str(path))
    metadata = pil_data.text
    data = np.asarray(pil_data)
    data = einops.rearrange(data, 'H W C -> C H W', C=4)
    flow, occlusions, uncertainty = data[:2, :, :], data[2, :, :], data[3, :, :]
    occlusions = einops.rearrange(occlusions, 'H W -> 1 H W')
    uncertainty = einops.rearrange(uncertainty, 'H W -> 1 H W')
    flow = decode(flow, float(metadata['flow_min']), float(metadata['flow_max']))
    occlusions = decode(occlusions, float(metadata['occl_min']), float(metadata['occl_max']))
    uncertainty = decode(uncertainty, float(metadata['unc_min']), float(metadata['unc_max']))
    return flow, occlusions, uncertainty


def write_flowou_X32(path, flow, occlusions, uncertainty):
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)
        uint32_xs = np.uint32(xs_01 * (2**32 - 1))
        return uint32_xs, lb, ub

    def u32_to_4u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x000000FF)
        byte_2 = np.uint8((xs & 0x0000FF00) >> 8)
        byte_3 = np.uint8((xs & 0x00FF0000) >> 16)
        byte_4 = np.uint8((xs & 0xFF000000) >> 24)
        return np.dstack((byte_4, byte_3, byte_2, byte_1))

    def encode_channel(xs):
        compressed_xs, lb, ub = compress_channel(xs)
        xs_4u8 = u32_to_4u8(compressed_xs)
        is_success, buf = cv2.imencode(".png", xs_4u8)
        # https://stackoverflow.com/a/52865864/1705970
        return {'data': buf,
                'min': lb,
                'max': ub}

    result = {
        'flow_x': encode_channel(flow[0, :, :]),
        'flow_y': encode_channel(flow[1, :, :]),
        'occlusion': encode_channel(occlusions[0, :, :]),
        'sigma': encode_channel(uncertainty[0, :, :])
        }

    with open(path, 'wb') as fout:
        pickle.dump(result, fout)


def read_flowou_X32(path):
    def decode_channel(data):
        buf = data['data']
        # https://stackoverflow.com/a/52865864/1705970
        xs_4u8 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        xs_compressed = data_4u8_to_u32(xs_4u8)
        xs = decompress_channel(xs_compressed, data['min'], data['max'])
        return xs

    def data_4u8_to_u32(xs):
        assert xs.dtype == np.uint8
        byte_4, byte_3, byte_2, byte_1 = np.dsplit(np.uint32(xs), 4)

        u32 = (byte_4 << 24) | (byte_3 << 16) | (byte_2 << 8) | byte_1
        return einops.rearrange(u32, 'H W 1 -> H W')

    def decompress_channel(compressed_xs, lb, ub):
        xs_01 = np.float32(compressed_xs) / (2**32 - 1)
        xs = (xs_01 * (ub - lb)) + lb
        return xs

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    flow_x = decode_channel(data['flow_x'])
    flow_y = decode_channel(data['flow_y'])
    flow = np.stack((flow_x, flow_y), axis=0)
    uncertainty = einops.rearrange(decode_channel(data['sigma']), 'H W -> 1 H W')
    occlusions = einops.rearrange(decode_channel(data['occlusion']), 'H W -> 1 H W')

    return flow, occlusions, uncertainty


def write_flowou_stepan16(path, flow, occlusions, uncertainty):
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)
        uint16_xs = np.uint16(np.round(xs_01 * (2**16 - 1)))
        return uint16_xs, lb, ub

    def u16_to_3u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x00FF)
        byte_2 = np.uint8((xs & 0xFF00) >> 8)
        byte_3 = np.uint8((xs & 0x0000) >> 16)
        return np.dstack((byte_3, byte_2, byte_1))

    def encode_channel(xs):
        compressed_xs, lb, ub = compress_channel(xs)
        xs_3u8 = u16_to_3u8(compressed_xs)
        is_success, buf = cv2.imencode(".png", xs_3u8)
        return {'data': buf,
                'min': lb,
                'max': ub}

    result = {
        'flow_x': encode_channel(flow[0, :, :]),
        'flow_y': encode_channel(flow[1, :, :]),
        'occlusion': encode_channel(occlusions[0, :, :]),
        'sigma': encode_channel(uncertainty[0, :, :])
        }

    path = str(path)
    suffix = '.stepan16'
    assert path.endswith(suffix)
    path = path[:-len(suffix)]

    flow_x_path = path + '_flow_x.png'
    flow_y_path = path + '_flow_y.png'
    cv2.imwrite(flow_x_path, result['flow_x']['data'])
    cv2.imwrite(flow_y_path, result['flow_y']['data'])
    limits_path = path + '_limits.txt'
    with open(limits_path, 'w') as fout:
        fout.write(f"{result['flow_x']['min']} {result['flow_x']['max']} {result['flow_y']['min']} {result['flow_y']['max']}")


def write_flowou_X16(path, flow, occlusions, uncertainty):
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)
        uint16_xs = np.uint16(np.round(xs_01 * (2**16 - 1)))
        return uint16_xs, lb, ub

    def u16_to_3u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x00FF)
        byte_2 = np.uint8((xs & 0xFF00) >> 8)
        byte_3 = np.uint8((xs & 0x0000) >> 16)
        return np.dstack((byte_3, byte_2, byte_1))

    def encode_channel(xs):
        compressed_xs, lb, ub = compress_channel(xs)
        xs_3u8 = u16_to_3u8(compressed_xs)
        is_success, buf = cv2.imencode(".png", xs_3u8)
        return {'data': buf,
                'min': lb,
                'max': ub}

    result = {
        'flow_x': encode_channel(flow[0, :, :]),
        'flow_y': encode_channel(flow[1, :, :]),
        'occlusion': encode_channel(occlusions[0, :, :]),
        'sigma': encode_channel(uncertainty[0, :, :])
        }

    with open(path, 'wb') as fout:
        pickle.dump(result, fout)


def read_flowou_X16(path):
    def decode_channel(data):
        buf = data['data']
        xs_3u8 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        xs_compressed = data_3u8_to_u16(xs_3u8)
        xs = decompress_channel(xs_compressed, data['min'], data['max'])
        return xs

    def data_3u8_to_u16(xs):
        assert xs.dtype == np.uint8
        byte_3, byte_2, byte_1 = np.dsplit(np.uint16(xs), 3)

        u16 = (byte_2 << 8) | byte_1
        return einops.rearrange(u16, 'H W 1 -> H W')

    def decompress_channel(compressed_xs, lb, ub):
        xs_01 = np.float32(compressed_xs) / (2**16 - 1)
        xs = (xs_01 * (ub - lb)) + lb
        return xs

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    flow_x = decode_channel(data['flow_x'])
    flow_y = decode_channel(data['flow_y'])
    flow = np.stack((flow_x, flow_y), axis=0)
    uncertainty = einops.rearrange(decode_channel(data['sigma']), 'H W -> 1 H W')
    occlusions = einops.rearrange(decode_channel(data['occlusion']), 'H W -> 1 H W')

    return flow, occlusions, uncertainty


class GeneralVideoCapture(object):
    """A cv2.VideoCapture replacement, that can also read images in a directory"""

    def __init__(self, path, reverse=False):
        images = Path(path).is_dir()
        self.image_inputs = images
        if images:
            self.path = path
            self.images = sorted([f for f in next(os.walk(path))[2]
                                  if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']])
            if reverse:
                self.images = self.images[::-1]
            self.i = 0
        else:
            self.cap = cv2.VideoCapture(str(path))

    def read(self):
        if self.image_inputs:
            if self.i >= len(self.images):
                return False, None
            img_path = os.path.join(self.path,
                                    self.images[self.i])
            self.frame_src = self.images[self.i]
            img = cv2.imread(img_path)
            self.i += 1
            return True, img
        else:
            return self.cap.read()

    def release(self):
        if self.image_inputs:
            return None
        else:
            return self.cap.release()


def get_video_frames(path):
    cap = GeneralVideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            return None
        yield frame


def get_video_length(path):
    N = 0
    for frame in get_video_frames(path):
        N += 1
    return N


class FlowCache():
    def __init__(self, cache_dir, max_RAM_MB=10000, max_GPU_RAM_MB=5000):
        self.cache_dir = cache_dir
        self.max_RAM_MB = max_RAM_MB
        self.max_GPU_RAM_MB = max_GPU_RAM_MB
        self.ram_cache = {}
        self.gpu_ram_cache = {}
        self.bytes_used = 0
        self.gpu_ram_bytes_used = 0
        self.n_saved = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _put(self, key, value):
        for tensor in value:
            self.bytes_used += sys.getsizeof(tensor.storage())
        self.ram_cache[key] = value

    def _put_gpu_ram(self, key, value):
        for tensor in value:
            self.gpu_ram_bytes_used += sys.getsizeof(tensor.storage())
        self.gpu_ram_cache[key] = value

    def _get(self, key):
        return self.ram_cache[key]

    def _get_gpu_ram(self, key):
        return self.gpu_ram_cache[key]

    def ram_space_left(self):
        max_bytes = self.max_RAM_MB * 1000000
        return max(max_bytes - self.bytes_used, 0)

    def gpu_ram_space_left(self):
        max_bytes = self.max_GPU_RAM_MB * 1000000
        return max(max_bytes - self.gpu_ram_bytes_used, 0)

    # @profile
    def read(self, left_id, right_id):
        key = (left_id, right_id)
        flow_left_to_right, occlusions, sigmas = None, None, None
        if key in self.gpu_ram_cache:
            flow_left_to_right, occlusions, sigmas = self._get_gpu_ram(key)
        elif key in self.ram_cache:
            flow_left_to_right, occlusions, sigmas = self._get(key)
            flow_left_to_right = flow_left_to_right.to('cuda')
            occlusions = occlusions.to('cuda')
            sigmas = sigmas.to('cuda')
        else:
            try:
                cache_path = self.cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
                assert cache_path.exists()
                flow_left_to_right, occlusions, sigmas = read_flowou(cache_path)
                flow_left_to_right = torch.from_numpy(flow_left_to_right).to('cuda')
                occlusions = torch.from_numpy(occlusions).to('cuda')
                sigmas = torch.from_numpy(sigmas).to('cuda')
                # when reading from disk, try to cache to GPU / RAM
                self.write(left_id, right_id, flow_left_to_right, occlusions, sigmas)  
            except Exception:
                pass

        return flow_left_to_right, occlusions, sigmas

    # @profile
    def write(self, left_id, right_id, flow_left_to_right, occlusions, sigmas):
        if self.gpu_ram_space_left() > 0:
            key = (left_id, right_id)
            self._put_gpu_ram(key, (flow_left_to_right, occlusions, sigmas))
        elif self.ram_space_left() > 0:
            key = (left_id, right_id)
            self._put(key, (flow_left_to_right.cpu(),
                            occlusions.cpu(),
                            sigmas.cpu()))
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path,
                             ensure_numpy(flow_left_to_right),
                             ensure_numpy(occlusions),
                             ensure_numpy(sigmas))
        self.n_saved += 1

    def clear(self, clear_disk=True):
        logger.debug(f'Saved {self.n_saved} flows, '
                     f'{len(self.gpu_ram_cache)} on GPU ({self.gpu_ram_bytes_used / 2**30:.2f}GiB), '
                     f'{len(self.ram_cache)} on RAM ({self.bytes_used / 2**30:.2f}GiB)')
        c = Counter()
        for left_id, right_id in self.ram_cache.keys():
            delta = abs(left_id - right_id)
            c[delta] += 1
        logger.debug(f'delta frequency: {c}')

        self.gpu_ram_cache.clear()
        self.gpu_ram_bytes_used = 0
        self.ram_cache.clear()
        self.bytes_used = 0
        self.n_saved = 0

        if clear_disk:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def backup_to_disk(self):
        """Save all the cached flowous to disk"""
        n_saved = 0
        for (left_id, right_id), val in tqdm.tqdm(list(self.ram_cache.items()), desc='saving RAM cache'):
            cache_path = self.cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path, *[ensure_numpy(x) for x in val])
                n_saved += 1

        for (left_id, right_id), val in tqdm.tqdm(list(self.gpu_ram_cache.items()), desc='saving GPU cache'):
            cache_path = self.cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path, *[ensure_numpy(x) for x in val])
                n_saved += 1
        logger.info(f"Saved {n_saved} cached flowous to disk.")

    def load_from_disk(self):
        all_cached = sorted(list(self.cache_dir.glob('*.flowouX16.pkl')))
        n_loaded = 0
        for path in tqdm.tqdm(all_cached, desc="loading flowous from disk"):
            left_id, right_id = Path(path.stem).stem.split('--')
            left_id, right_id = int(left_id), int(right_id)

            try:
                flow_left_to_right, occlusions, sigmas = read_flowou(path)
                flow_left_to_right = torch.from_numpy(flow_left_to_right).to('cuda')
                occlusions = torch.from_numpy(occlusions).to('cuda')
                sigmas = torch.from_numpy(sigmas).to('cuda')
                self.write(left_id, right_id, flow_left_to_right, occlusions, sigmas)
                n_loaded += 1
            except Exception:
                pass
        logger.info(f"Loaded {n_loaded} flowous into cache.")
