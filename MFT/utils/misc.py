import numpy as np
import torch


def try_intize(x):
    try:
        if str(int(x)) == x:
            return int(x)
    except Exception:
        pass
    return x


def remap(x,
          src_min, src_max,
          dst_min, dst_max):
    x_01 = (x - src_min) / np.float64(src_max - src_min)
    x_dst = x_01 * (dst_max - dst_min) + dst_min

    return x_dst


def ensure_numpy(xs):
    if isinstance(xs, torch.Tensor):
        return xs.detach().cpu().numpy()
    else:
        return xs


def ensure_torch(xs):
    if not isinstance(xs, torch.Tensor):
        return torch.from_numpy(xs)
    else:
        return xs


def make_pairs(xs):
    """Given an iterable xs, generate consecutive (overlapping) pairs of items from x.

    With xs = [1, 2, 3, 4, 5],
    generate (1, 2), (2, 3), (3, 4), (4, 5)"""
    last_x = None
    last_x_set = False
    for x in xs:
        if not last_x_set:
            last_x = x
            last_x_set = True
            continue
        else:
            yield (last_x, x)
            last_x = x


def make_delta_pairs(xs, delta):
    """ make pairs (i-delta, i) """
    xs = list(xs)
    for i, x in enumerate(xs):
        left_i = i - delta
        if left_i < 0 or left_i >= len(xs):
            continue

        yield (left_i, xs[left_i], i, x)


def parse_scale_WH(scale_WH, frames_shape):
    """ Parse string argument scale_WH and scale img_shape according to scale_WH.
        Missing value will be computed keeping same ratio.

    Params: scale_WH       - string with resolution WxH, examples: fullres, 256x256, x1080, 512x,
                             if there is "_" in the scale_WH, the output will be sequence of rescaling operations:
                             i.e. 256x256_x480 -> first rescale to 256x256 then to x480
            frames_shape   - dict with current's frame resolution with keys W and H
    Output: new_shape_list - list of dicts with scaled resolution, video will be rescaled multiple times according
                             the new resolutions in list
    """
    if scale_WH == 'fullres':
        return [frames_shape]
    new_shape_list = []
    scale_WH_split = scale_WH.split('_')
    for c_scale_WH in scale_WH_split:
        if c_scale_WH == 'fullres':
            new_shape_list.append(frames_shape)
            continue
        new_shape = dict(frames_shape.items())
        W_str, H_str = c_scale_WH.split('x')
        W = int(W_str) if W_str != '' else None
        H = int(H_str) if H_str != '' else None
        assert W is not None or H is not None, 'at least one dimmension has to be set'
        new_shape['W'] = W if W is not None else int(round(frames_shape['W'] * (H / frames_shape['H'])))
        new_shape['H'] = H if H is not None else int(round(frames_shape['H'] * (W / frames_shape['W'])))
        new_shape_list.append(new_shape)
    return new_shape_list


def trim_string(x, max_len, end="..."):
    assert len(end) < max_len
    if len(x) > max_len:
        x = x[:max_len - len(end)] + end

    return x


def align_string(x, align, width, fill_char=None):
    assert align in ['>', '<', '^']
    if fill_char is None:
        fill_char = ''
    aligned = '{x:{fill_char}{align}{width}}'.format(
        x=x, align=align, width=width, fill_char=fill_char)
    return aligned
