import numpy as np
import torch
import torch.nn.functional as F
import einops

from scipy.interpolate import RegularGridInterpolator
from MFT.utils.geom_utils import torch_get_featuremap_coords


def chain_flow(flow_AB, flow_BC):
    """ chain flow by bilinear interpolation """
    if flow_AB is None:
        return flow_BC

    flow_shape = einops.parse_shape(flow_AB, 'batch delta H W')
    coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_AB.device,
                                           keep_shape=True)
    coords_B = coords_A + flow_AB

    coords_C = bilinear_interpolate_torch(
        flow_BC,
        einops.rearrange(coords_B, 'xy H W -> xy (H W)', xy=2)[1, :],
        einops.rearrange(coords_B, 'xy H W -> xy (H W)', xy=2)[0, :])
    print(f"coords_C.shape: {coords_C.shape}")


def chain_flow_single(flow_AB, coords_A=None, return_flow=False):
    """ Chains flow by bilinear interpolation

    args:
        flow_AB: (N, xyDelta, H, W) tensor
        coords_A: [optional] (1, xy, H, W) tensor of original grid coordinates
        return_flow: [optional, default=False] return flow from A to B, instead of B coords

    returns:
        coords_B: (xy, H, W) tensor of resulting grid coordinates (original coordinates warped by flow)
    """
    device = flow_AB.device
    if coords_A is None:
        flow_shape = einops.parse_shape(flow_AB, 'batch delta H W')
        coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=device,
                                               keep_shape=True)
        coords_A = einops.rearrange(coords_A, '(N xy) H W -> N H W xy', xy=2, N=1)

    # the (x, y) coordinates should be normalized such that top-left is (-1, -1) and bottom-right is (1, 1)
    H, W = flow_AB.shape[2], flow_AB.shape[3]
    coords_A_normed = normalize_coords(coords_A, H, W)
    coords_B_delta = F.grid_sample(flow_AB,
                                   # torch.flip(coords_A_normed, dims=[3]),
                                   coords_A_normed,
                                   align_corners=True)
    if return_flow:
        coords_B = coords_B_delta
    else:
        coords_B = einops.rearrange(coords_A,
                                    'N H W xy -> N xy H W', xy=2) + coords_B_delta

    coords_B = einops.rearrange(coords_B, 'N xy H W -> (N xy) H W', N=1, xy=2)

    return coords_B


def normalize_coords(coordinates, H, W):
    """Normalize coordinates to be in [-1, 1] range as usedi in F.grid_sample.
    args:
        coordinates: (N H W xy) coordinates
    """
    device = coordinates.device
    scales = np.array([2 / (W - 1), 2 / (H - 1)]).astype(np.float32)  # maps to [0, 2]
    scales = torch.from_numpy(scales).to(device)
    scales = einops.rearrange(scales, '(N H W xy) -> N H W xy', N=1, H=1, W=1)
    coordinates_normed = coordinates * scales - 1      # maps to [-1, 1]
    return coordinates_normed


def bilinear_sample(data, coords, device=None):
    """
    args:
        data: (batch, C, H, W) tensor
        coords: (batch, ...outshape..., xy) tensor of coordinates
    returns:
        sampled: (batch, ...outshape..., C) tensor
    """
    assert len(coords.shape) >= 3
    assert coords.shape[-1] == 2
    if device is None:
        device = coords.device
    data_shape = einops.parse_shape(data, 'batch C H W')
    norm_coords = normalize_coords(coords.to(device), data_shape['H'], data_shape['W'])
    flat_coords = norm_coords.view(coords.shape[0], -1, coords.shape[-1])  # flatten all the outshape
    grid_coords = einops.rearrange(flat_coords, 'batch N xy -> batch 1 N xy', xy=2)  # simulate HxW coord grid (but only 1xN)
    sampled_flat = F.grid_sample(data.to(device), grid_coords, align_corners=True)
    sampled = sampled_flat.view(*coords.shape[:-1], data_shape['C'])
    return sampled


def chain_flow_2(flow_AB, flow_BC, coords_A=None):
    """ Chains flow by bilinear interpolation

    args:
        flow_AB: (N, xyDelta, H, W) tensor
        flow_BC: (N, xyDelta, H, W) tensor
        coords: [optional] (1, xy, H, W) tensor of original grid coordinates
    """
    if coords_A is None:
        flow_shape = einops.parse_shape(flow_AB, 'batch delta H W')
        coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_AB.device,
                                               keep_shape=True)

    coords_B = coords_A + flow_AB
    # the (x, y) coordinates should be normalized such that top-left is (-1, -1) and bottom-right is (1, 1)
    H, W = flow_AB.shape[[2, 3]]
    scales = np.array([2 / (W - 1), 2 / (H - 1)])  # maps to [0, 2]
    scales = einops.rearrange(scales, 'xy -> N xy H W', N=1, H=1, W=1)
    coords_B_normed = coords_B * scales - 0.5      # maps to [-1, 1]
    coords_C = F.grid_sample(flow_BC, coords_B_normed)

    return coords_C


class FlowInterpolator(object):
    def __init__(self, flow, additional_data=None):
        """
            flow: (H, W, 2) array of dx, dy pairs
            additional_data: (H, W, [C]) array.
        """
        H, W, C = flow.shape
        assert C == 2
        flow_grid_ys = np.arange(H)
        flow_grid_xs = np.arange(W)
        if additional_data is None:
            data = flow
        else:
            if len(additional_data.shape) < 3:
                additional_data = additional_data[:, :, np.newaxis]
            data = np.concatenate((flow, additional_data), axis=2)
        self.interp = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), data, bounds_error=False, fill_value=np.nan)

    def __call__(self, positions, method='linear'):
        """
        args:
                positions: (N, 2) array of x, y pairs (possibly non-integer)
        """
        return self.interp(positions[:, ::-1], method=method)  # the scipy interpolator wants y, x coordinates


def interp_flow(current_positions, flow, occlusion_mask=None):
    """Interpolate flow and occlusion masks from non-integer starting point

    args:
        current_positions: (N, 2) array of y, x pairs (non-integer)
        flow: (H, W, 2) array of dx, dy pairs
        occlusion_mask: (H, W) array. Values > 0 mean that the pixel in left image is occluded in the right image
    """
    H, W, C = flow.shape
    assert C == 2

    flow_grid_ys = np.arange(H)
    flow_grid_xs = np.arange(W)
    interp_flow = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), flow, bounds_error=False, fill_value=np.nan)
    new_flow = interp_flow(current_positions, method='linear')

    if occlusion_mask is not None:
        interp_occl = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), occlusion_mask, bounds_error=False, fill_value=1)
        new_occl = interp_occl(current_positions, method='linear')
    else:
        new_occl = None

    return new_flow, new_occl


def flow_warp_coords(coords_A, flow_AB):
    """warp (non-integer) coordinates coords_A using optical flow.

    args:
        coords_A: (2, N) tensor with x, y coordinates
        flow_AB: (2, H, W) tensor with optical flow

    returns:
        coords_B: (2, N) tensor with x, y coordinates
    """
    sampled_flow = bilinear_interpolate_torch(
        flow_AB,
        coords_A[0, :],
        coords_A[1, :])
    coords_B = coords_A + sampled_flow
    return coords_B


def bilinear_interpolate_torch(data, x, y):
    """
    Bilinear interpolation of (CxHxW) im tensor

    args:
        data: (C, H, W) tensor with data to be interpolated
        x: (N, ) tensor with x coordinates into data
        y: (N, ) tensor with y coordinates into data
    returns:
        interp: (C, N) tensor with values interpolated from data

    https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    """
    H, W = data.shape[1], data.shape[2]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    data_a = data[:, y0, x0]
    data_b = data[:, y1, x0]
    data_c = data[:, y0, x1]
    data_d = data[:, y1, x1]

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_a = einops.rearrange((x1 - x) * (y1 - y), 'N -> 1 N')
    w_b = einops.rearrange((x1 - x) * (y - y0), 'N -> 1 N')
    w_c = einops.rearrange((x - x0) * (y1 - y), 'N -> 1 N')
    w_d = einops.rearrange((x - x0) * (y - y0), 'N -> 1 N')

    interp = (w_a * data_a) + (w_b * data_b) + (w_c * data_c) + (w_d * data_d)
    return interp


def bilinear_splat(data, data_coords, grid_shape):
    """
    Bilinear splat the data (at data_coords) onto a grid

    args:
        data: (N, C) tensor with data to be splatted
        data_coords: (N, 2) tensor with xy coordinates of the data
        grid_shape: (2, ) tuple of (height, width) of the splatting grid
    returns:
        grid: (H, W, C) tensor with splatted data
        counts: (H, W, 1) tensor with splatted ones
    """
    assert data.device == data_coords.device
    device = data.device

    H, W = grid_shape[:2]
    C = data.shape[1]

    x = data_coords[:, 0]
    y = data_coords[:, 1]

    ## find the surrounding grid positions
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # stay inside the grid
    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # compute the bilinear coefficients
    x0f = x0.float()
    x1f = x1.float()
    y0f = y0.float()
    y1f = y1.float()
    w_a = einops.rearrange((x1f - x) * (y1f - y), 'N -> N 1')
    w_b = einops.rearrange((x1f - x) * (y - y0f), 'N -> N 1')
    w_c = einops.rearrange((x - x0f) * (y1f - y), 'N -> N 1')
    w_d = einops.rearrange((x - x0f) * (y - y0f), 'N -> N 1')

    # assert torch.all(w_a >= 0)
    # assert torch.all(w_b >= 0)
    # assert torch.all(w_c >= 0)
    # assert torch.all(w_d >= 0)
    # assert torch.all(w_a <= 1)
    # assert torch.all(w_b <= 1)
    # assert torch.all(w_c <= 1)
    # assert torch.all(w_d <= 1)

    row_indices = torch.cat((y0, y1, y0, y1), dim=0)
    col_indices = torch.cat((x0, x0, x1, x1), dim=0)
    flat_indices = torch_ravel_multi_index((row_indices, col_indices), (H, W))
    flat_data = torch.cat((data * w_a, data * w_b, data * w_c, data * w_d), dim=0)
    # flat_data = einops.rearrange(flat_data, 'N 1 -> N')
    flat_count = torch.cat((w_a, w_b, w_c, w_d), dim=0)

    grid_flat = torch.zeros((H * W, C), dtype=flat_data.dtype, device=device)
    grid_flat = grid_flat.index_put(indices=[flat_indices], values=flat_data, accumulate=True)

    counts_flat = torch.zeros((H * W, 1), dtype=flat_count.dtype, device=device)
    counts_flat = counts_flat.index_put(indices=[flat_indices], values=flat_count, accumulate=True)
    # data_a = data[:, y0, x0]
    # data_b = data[:, y1, x0]
    # data_c = data[:, y0, x1]
    # data_d = data[:, y1, x1]

    # interp = (w_a * data_a) + (w_b * data_b) + (w_c * data_c) + (w_d * data_d)
    grid = einops.rearrange(grid_flat, '(H W) C -> H W C', H=H, W=W, C=C)
    counts = einops.rearrange(counts_flat, '(H W) 1 -> H W 1', H=H, W=W)
    return grid, counts


def torch_ravel_multi_index(multi_index, dims):
    """Converts a tuple of index arrays into an array of flat indices.

    A counterpart to numpy ravel_multi_index.

    args:
        multi_index: a tuple of (N, ) integer tensors, one for each dimension
        dims: the shape of the array that multi_index indices refer to
    returns:
        raveled_indices: a (N, ) tensor of flat indices
    """
    if len(dims) != 2:
        raise NotImplementedError("Too lazy to do the most general thing... :)")
    H, W = dims
    rows = multi_index[0]
    cols = multi_index[1]

    raveled_indices = W * rows + cols
    return raveled_indices


def forward_backward_error(flow_forward, flow_backward):
    """
    args:
        flow_forward: (xy H W) tensor with flow from A to B
        flow_backward: (xy H W) tensor with flow from B to A
    return:
        error: (xy H W) tensor with forward backward A->B->A error
    """
    device = flow_forward.device
    flow_shape = einops.parse_shape(flow_forward, 'xy H W')
    assert flow_shape['xy'] == 2

    coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']),
                                           device=device, keep_shape=True)
    coords_B = coords_A + flow_forward.to(device).to(torch.float32)
    coords_B_normed = normalize_coords(
        einops.rearrange(coords_B, 'xy H W -> 1 H W xy', xy=2),
        flow_shape['H'], flow_shape['W'])

    flow_BA = flow_backward.to(torch.float32)
    flow_BA_sampled = F.grid_sample(
        einops.rearrange(flow_BA, 'xy H W -> 1 xy H W', xy=2),
        coords_B_normed,
        align_corners=True)
    flow_ABA = coords_B + flow_BA_sampled - coords_A
    error = einops.rearrange(flow_ABA, '1 xy H W -> xy H W', xy=2)
    return error


def forward_backward_error_magnitude(flow_forward, flow_backward):
    """
    args:
        flow_forward: (xy H W) tensor with flow from A to B
        flow_backward: (xy H W) tensor with flow from B to A
    return:
        error_magnitude: (H W) tensor with forward backward A->B->A error magnitude
    """
    flow_ABA = forward_backward_error(flow_forward, flow_backward)
    error_magnitude = torch.sqrt(
        einops.reduce(torch.square(flow_ABA),
                      'xy H W -> H W', xy=2,
                      reduction='sum'))
    return error_magnitude
