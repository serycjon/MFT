import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
except ImportError:
    # print('geomutils torch import error')
    pass
from functools import reduce
from collections import defaultdict, deque
import scipy.linalg
import cv2
import einops
from types import SimpleNamespace


class Bbox:
    def __init__(self, tl_x=None, tl_y=None, w=None, h=None):
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.w = w
        self.h = h

        self.br_x = self.tl_x + self.w - 1
        self.br_y = self.tl_y + self.h - 1

    def __repr__(self):
        return f"Bbox(tl_x={self.tl_x}, tl_y={self.tl_y}, w={self.w}, h={self.h})"

    @classmethod
    def from_xyxy(cls, xyxy):
        tl_x = xyxy[0]
        tl_y = xyxy[1]
        br_x = xyxy[2]
        br_y = xyxy[3]
        w = br_x - tl_x + 1
        h = br_y - tl_y + 1

        bbox = cls(tl_x, tl_y, w, h)
        return bbox

    @classmethod
    def from_xywh(cls, xywh):
        bbox = cls(*xywh)
        return bbox

    @classmethod
    def from_cxcywh(cls, cxcywh):
        cx, cy, w, h = cxcywh
        tl_x = cx - (w - 1) / 2
        tl_y = cy - (h - 1) / 2

        bbox = cls(tl_x, tl_y, w, h)
        return bbox

    @classmethod
    def from_mask(cls, binary_image):
        if not np.any(binary_image):
            return Bbox.from_xyxy((0, 0, 0, 0))

        # faster version, thanks to:
        # https://stackoverflow.com/a/31402351/1705970
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        tl_x = cmin
        tl_y = rmin
        br_x = cmax
        br_y = rmax

        bbox = Bbox.from_xyxy((tl_x, tl_y, br_x, br_y))
        return bbox

    @classmethod
    def from_points(cls, pts):
        """Create a bounding box from a bunch of points.

        args:
          pts: (2, N) x, y points
        """
        min_x, max_x = np.amin(pts[0, :]), np.amax(pts[0, :])
        min_y, max_y = np.amin(pts[1, :]), np.amax(pts[1, :])
        return cls.from_xyxy([min_x, min_y, max_x, max_y])

    def as_xyxy(self):
        return [self.tl_x, self.tl_y, self.br_x, self.br_y]

    def as_xywh(self):
        return [self.tl_x, self.tl_y, self.w, self.h]

    def as_points(self):
        return [[self.tl_x, self.tl_y],
                [self.br_x, self.tl_y],
                [self.br_x, self.br_y],
                [self.tl_x, self.br_y]]

    def get_center(self):
        return [self.tl_x + self.w // 2, self.tl_y + self.h // 2]

    def rounded_to_int(self):
        def round_to_int(x):
            return int(np.round(x))
        return Bbox.from_xyxy((round_to_int(self.tl_x),
                               round_to_int(self.tl_y),
                               round_to_int(self.br_x),
                               round_to_int(self.br_y)))

    def with_margins(self, margin_fraction):
        return Bbox.from_xyxy((self.tl_x - int(margin_fraction * self.w),
                               self.tl_y - int(margin_fraction * self.h),
                               self.br_x + int(margin_fraction * self.w),
                               self.br_y + int(margin_fraction * self.h)))

    def with_margins_min_size(self, min_w, min_h=None):
        if min_h is None:
            min_h = min_w

        missing_w = max(min_w - self.w, 0) / 2
        missing_h = max(min_h - self.h, 0) / 2
        missing_w_margin = missing_w / self.w
        missing_h_margin = missing_h / self.h
        missing_margin = max(missing_w_margin, missing_h_margin)
        if missing_margin > 0:
            bbox = self.with_margins(missing_margin)
        else:
            bbox = self
        return bbox

    def draw(self, canvas, color=(0, 0, 255), thickness=2):
        cv2.rectangle(canvas, (int(self.tl_x), int(self.tl_y)),
                      (int(self.br_x), int(self.br_y)),
                      color, thickness)
        return canvas

    def intersection(self, other):
        intersection_xyxy = [max(self.tl_x, other.tl_x),
                             max(self.tl_y, other.tl_y),
                             min(self.br_x, other.br_x),
                             min(self.br_y, other.br_y)]
        return Bbox.from_xyxy(intersection_xyxy)

    def crop_image(self, img):
        rounded = self.rounded_to_int()
        cropped = img[rounded.tl_y:rounded.br_y,
                      rounded.tl_x:rounded.br_x,
                      ...]
        return cropped

    def is_pt_inside(self, xy):
        return xy[0] > self.tl_x and xy[0] < self.br_x and \
            xy[1] > self.tl_y and xy[1] < self.tl_y

    def sample_img(self, img, target_box, interpolation=cv2.INTER_NEAREST):
        assert target_box.tl_x == 0 and target_box.tl_y == 0

        H = H_bbox2bbox(self, target_box)
        sample = cv2.warpPerspective(img, H,
                                     (target_box.w, target_box.h),
                                     flags=interpolation)
        return sample


def H_bbox2bbox(src, dst):
    ''' Compute homography mapping bounding boxes

    Args:
        src: Bbox
        dst: Bbox
    '''
    # homography contstructed by unshifting src box topleft to zero,
    # scaling to the dst box size and shifting to the crop box topleft
    H_unshift = np.eye(3)
    H_unshift[0, 2] = -src.tl_x  # unshift x
    H_unshift[1, 2] = -src.tl_y  # unshift y

    scale_h = dst.h / float(src.h)
    scale_w = dst.w / float(src.w)
    H_scale = np.diag((scale_w, scale_h, 1))

    H_shift = np.eye(3)
    H_shift[0, 2] = dst.tl_x  # shift x
    H_shift[1, 2] = dst.tl_y  # shift y

    # H operates on data from left -> reverse order
    H = np.matmul(H_shift, np.matmul(H_scale, H_unshift))
    H /= H[2, 2]
    return H


def max_fitting_bbox(src_bbox, target_bbox):
    w_scale = target_bbox.w / src_bbox.w
    h_scale = target_bbox.h / src_bbox.h

    scale = min(w_scale, h_scale)

    return Bbox(target_bbox.tl_x, target_bbox.tl_y, src_bbox.w * scale, src_bbox.h * scale)


def project_bbox(bbox, H):
    ''' Project bbox by homography

    Args:
        bbox: Bbox
        H: (3, 3) homography

    Returns:
        proj_bbox: Bbox
    '''

    ids = ((0, 1), (2, 1), (2, 3), (0, 3))
    xyxy_bbox = bbox.as_xyxy()
    x = np.array([[xyxy_bbox[x_id], xyxy_bbox[y_id]] for x_id, y_id in ids]).T
    assert x.shape == (2, 4)

    proj = p2e(np.matmul(H, e2p(x))).T
    proj_bbox = (proj[0, 0], proj[0, 1], proj[2, 0], proj[2, 1])
    proj_bbox = Bbox.from_xyxy(proj_bbox)
    return proj_bbox


def H_proj(H, points):
    """ Projects D-dimensional points by a homography

    args:
        H: (D+1, D+1) homography matrix
        points: (D, N) points to be projected

    returns:
        projected: (D, N) projected points
    """
    return p2e(np.matmul(H, e2p(points)))


def torch_H_proj(H, points):
    assert H.shape == (3, 3)
    assert points.shape[0] == 2
    assert len(points.shape) == 2
    assert H.dtype == torch.float64
    assert points.dtype == torch.float64
    N_points = points.shape[1]
    homo_points = torch.cat((points, torch.ones((1, N_points), dtype=torch.float64, device=points.device)),
                            dim=0)
    assert homo_points.shape == (3, N_points)
    homo_proj = torch.matmul(H, homo_points)
    proj = homo_proj[:2, :] / homo_proj[2:, :]

    # np_H = H.cpu().detach().numpy()
    # np_points = points.cpu().detach().numpy()
    # check = H_proj(np_H,
    #                np_points)
    # print(f"check: {check}")
    return proj


def e2p(xs):
    ''' converts (N, D) euclidean coordinates to (N, D+1) projective (homogenous coords) '''
    return np.vstack((xs, np.ones(xs.shape[1])))


def p2e(xs):
    ''' converts (N, D+1) homogenous coordinates to (N, D) euclidean '''
    assert xs.shape[1] >= 1
    return xs[:-1, :] / np.reshape(xs[-1, :], (1, xs.shape[1]))


def in_bounds(x, lb, ub, axis=None):
    return np.logical_and(
        np.all(x >= lb, axis=axis),
        np.all(x < ub, axis=axis))


def torch_in_bounds(x, lb, ub, axis=None):
    return torch.all(x >= lb, dim=axis) & torch.all(x < ub, dim=axis)


def cv_crop_bbox(img, bbox_xywh):
    bbox = Bbox.from_xywh(bbox_xywh)
    target = Bbox.from_xywh((0, 0, bbox.w, bbox.h))
    H = H_bbox2bbox(bbox, target)

    cropped = cv2.warpPerspective(img, H,
                                  (bbox.w, bbox.h),
                                  flags=cv2.INTER_NEAREST)
    return cropped


def crop_bbox_H(bbox):
    target = Bbox.from_xywh((0, 0, bbox.w, bbox.h))
    H = H_bbox2bbox(bbox, target)
    return H


def crop_bbox(img, src_bbox, out_H, out_W,
              with_padding=True):
    ''' Crop image by roialign

    Args:
        img: torch ([1, ]CH, H, W) tensor
        src_bbox: Bbox
    '''
    orig_shape = img.shape
    if len(orig_shape) == 3:
        img = img.view(1, *img.shape)

    _, CH, H, W = img.shape

    if with_padding:
        # pad the image if we are cropping outside of it
        pad_left = int(max(0 - src_bbox.tl_x, 0))
        pad_top = int(max(0 - src_bbox.tl_y, 0))
        pad_right = int(max(src_bbox.br_x - (W - 1), 0))
        pad_bottom = int(max(src_bbox.br_y - (H - 1), 0))

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom),
                    mode='replicate')

        src_bbox_roi = Bbox.from_xyxy((pad_left + src_bbox.tl_x,
                                       pad_top + src_bbox.tl_y,
                                       pad_left + src_bbox.br_x,
                                       pad_top + src_bbox.br_y))
    else:
        src_bbox_roi = src_bbox

    # finally crop the bbox
    align_bbox = [torch.tensor(src_bbox_roi.as_xyxy()).view(1, 4).float().cuda()]
    roi = roi_align(img, align_bbox, (out_H, out_W))
    if len(orig_shape) == 3:
        roi = roi.view(roi.shape[1:])

    # compute performed homography
    dst_bbox = Bbox.from_xywh((0, 0, out_W, out_H))
    H = H_bbox2bbox(src_bbox, dst_bbox)
    return roi, H


def polygon_orientations(poly, take=None):
    assert poly.shape[0] == 2
    assert poly.shape[1] >= 3
    N = poly.shape[1]
    if take is None:
        take = N - 1

    orientations = []
    for i in range(take):
        x_a, y_a = poly[:, i - 1]
        x_b, y_b = poly[:, i]
        x_c, y_c = poly[:, i + 1]

        # https://en.wikipedia.org/wiki/Curve_orientation
        ori = np.sign((x_b - x_a) * (y_c - y_a) - (x_c - x_a) * (y_b - y_a))
        orientations.append(ori)
    return np.array(orientations)


def compose_H(*Hs):
    """ Compose homographies (multiply in reverse order). """
    for H in Hs:
        if H is None:
            return None
    result = reduce(np.dot, reversed(Hs))
    result /= result[2, 2]  # normalize to 1 in bottomright

    return result


class HCoordSystemGraph:
    def __init__(self):
        self.nodes = defaultdict(dict)

    def add(self, src_name, dst_name, H_src2dst):
        result = self.copy()
        result.add_mutating(src_name, dst_name, H_src2dst)
        return result

    def add_mutating(self, src_name, dst_name, H_src2dst):
        self.nodes[src_name][dst_name] = H_src2dst.copy()
        H_dst2src = np.linalg.inv(H_src2dst)
        self.nodes[dst_name][src_name] = H_dst2src

    def get(self, src_name, dst_name):
        visited = [src_name]
        queue = deque([(src_name, np.eye(3))])

        while queue:
            current, H_src2current = queue.pop()
            if current == dst_name:
                return H_src2current

            for neighbor, H_cur2neighbor in self.nodes[current].items():
                if neighbor not in visited:
                    visited.append(neighbor)
                    H_src2neighbor = compose_H(H_src2current, H_cur2neighbor)
                    queue.append((neighbor, H_src2neighbor))
        plot_file = '/tmp/h_graph.gv'
        self.plot(plot_file)
        raise RuntimeError(f"No known transformation from {src_name} to {dst_name}. Check {plot_file}")

    def plot(self, out_path='/tmp/h_graph.gv'):
        from graphviz import Digraph
        g = Digraph()

        for src_name, node in self.nodes.items():
            for dst_name, _ in node.items():
                g.edge(src_name, dst_name)
        g.render(out_path, format='png')

    def copy(self):
        out = HCoordSystemGraph()
        for src_name, node in self.nodes.items():
            for dst_name, H in node.items():
                out.add_mutating(src_name, dst_name, H)
        return out


def A2H(A):
    if A is not None:
        assert A.shape == (2, 3)
        return np.concatenate((A, [[0, 0, 1]]), axis=0)


def H_interpolate(H_a, H_b, t):
    res = scipy.linalg.expm((1 - t) * scipy.linalg.logm(H_a) + t * scipy.linalg.logm(H_b))
    assert np.allclose(np.imag(res), 0)
    return np.real(res)


def torch_get_featuremap_coords(feature_map, device=None,
                                keep_shape=False):
    """ get coordinate map corresponding to a feature map

    args:
        feature_map: (..., H, W) tensor
        keep_shape: boolean (default False). Setting it to True does not flatten the output coordinates

    returns:
        xy: (2, H*W) tensor with x, y coordinates.  (2, H, W) tensor if keep_shape == True
    """
    if type(feature_map) is tuple and len(feature_map) == 2:
        H, W = feature_map
        assert device is not None
    else:
        H, W = feature_map.shape[-2:]
        if device is None:
            device = feature_map.device
    xy = unravel_indices(torch.arange(H * W, device=device), (H, W), stack_dim=0)

    if keep_shape:
        xy = einops.rearrange(xy, 'xy (H W) -> xy H W',
                              H=H, W=W, xy=2)
    return xy


def get_featuremap_coords(feature_map):
    """ get coordinate map corresponding to a feature map

    args:
        feature_map: (..., H, W) np array or tensor, or tuple (H, W)

    returns:
        xy: (2, H*W) np array or tensor with x, y coordinates
    """
    if type(feature_map) is tuple and len(feature_map) == 2:
        H, W = feature_map
    else:
        H, W = feature_map.shape[-2:]
    ys, xs = np.unravel_index(np.arange(H * W), (H, W))
    xy = np.stack((xs, ys), axis=0)
    return xy


def unravel_indices(indices, shape, stack_dim=-1, np_order=False):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).

    From https://github.com/pytorch/pytorch/issues/35674#issuecomment-739560051
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    if np_order:  # row, column (y, x)
        coord = torch.stack(coord[::-1], dim=stack_dim)
    else:  # column, row (x, y)
        coord = torch.stack(coord, dim=stack_dim)
    assert coord.device == indices.device

    return coord


def find_TRS(left_coords, right_coords):
    """Estimate translation, rotation, scale transformation mapping from left to right coordinates
    The TRS transformation is estimated by least squares.

    Rotation matrix: [[cos, -sin], [sin, cos]]
    Scaled rotation: [[A, B], [-B, A]]
    TRS: [[A, B, C], [-B, A, D]]

    TRS * [X, Y, 1]^T = [[A*X + B*Y    + C*1 + D*0],
                         [A*Y + B*(-X) + C*0 + D*1]]

    Each correspondence thus adds into the system:
    [X,  Y, 1, 0]                     [X']
    [Y, -X, 0, 1]  * [A, B, C, D]^T = [Y']

    args:
        left_coords: (N, delta-xy) float tensor
        right_coords: (N, delta-xy) float tensor
    returns:
        trans: (2, 3) TRS transformation matrix tensor
    """
    X, Y = torch.chunk(left_coords, dim=1, chunks=2)
    X_prime, Y_prime = torch.chunk(right_coords, dim=1, chunks=2)
    ones, zeros = torch.ones_like(X), torch.zeros_like(X)

    ax = torch.cat([X,  Y, ones,  zeros], dim=1)
    ay = torch.cat([Y, -X, zeros, ones], dim=1)

    A = torch.cat((ax, ay), dim=1)  # each correspondence crammed into one row
    b = torch.cat((X_prime, Y_prime), dim=1)
    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(A, 'N (two four) -> 1 (N two) four', two=2, four=4)  # add batch
    b = einops.rearrange(b, 'N (two one) -> 1 (N two) one', two=2, one=1)  # add batch

    # now we just have to solve Ax = b (where x = [A, B, C, D])
    res = torch.linalg.qr(A)

    # now we need to solve Rx = Q^T b
    # res.Q has shape (batch, 2xN, 4)
    # N = left_coords.shape[0]
    # assert res.Q.shape == (1, 2 * N, 4)

    lhs = res.R
    rhs = einops.rearrange(res.Q, 'B twoN four -> B four twoN', four=4) @ b

    res = torch.triangular_solve(rhs, lhs)
    solution = res.solution  # batch, 4, 1

    A, B, C, D = torch.chunk(solution, dim=1, chunks=4)

    trans_x = torch.cat((A, B, C), dim=2)
    trans_y = torch.cat((-B, A, D), dim=2)

    trans = torch.cat((trans_x, trans_y), dim=1)
    return einops.rearrange(trans, '1 rows cols -> rows cols', rows=2, cols=3)


def flow_to_TRS_flow(flow, mask=None):
    """Given a (H, W, xy-delta) flow, estimate Translation, Rotation,
    Scale (TRS) transformation and convert it back to flow.  The TRS
    transformation is estimated by least squares.

    args:
        flow: (H, W, xy-delta) tensor
        mask: [optional] (H, W) binary np.array or tensor (selects flow vectors to be used for TRS estimation)
    """
    flow_shape = einops.parse_shape(flow, 'H W xy')
    TRS, aux = flow_to_TRS(flow, mask)
    TRS_flow = Affine_to_flow(TRS, aux.left_coords, flow_shape)
    return TRS_flow


def flow_to_TRS(flow, mask=None):
    """Given a (H, W, xy-delta) flow, estimate Translation, Rotation,
    Scale (TRS) transformation.  The TRS
    transformation is estimated by least squares.

    args:
        flow: (H, W, xy-delta) tensor
        mask: [optional] (H, W) binary np.array or tensor (selects flow vectors to be used for TRS estimation)
    """
    # Convert flow to correspondences:
    flow_shape = einops.parse_shape(flow, 'H W xy')
    device = flow.device
    grid_left_img = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']),
                                                device=device, keep_shape=True)
    grid_right_img = einops.rearrange(grid_left_img, 'xy H W -> H W xy', xy=2) + flow

    left_coords = einops.rearrange(grid_left_img, 'xy H W -> (H W) xy', xy=2).to(torch.float32)
    right_coords = einops.rearrange(grid_right_img, 'H W xy -> (H W) xy', xy=2).to(torch.float32)

    if mask is not None:
        flat_mask = einops.rearrange(mask, 'H W -> (H W)')

        src = left_coords[flat_mask, :]
        dst = right_coords[flat_mask, :]
    else:
        src = left_coords
        dst = right_coords

    TRS = find_TRS(src, dst)
    aux = SimpleNamespace()
    aux.left_coords = left_coords
    return TRS, aux


def Affine_to_flow(A, left_coords, flow_shape):
    to_proj = torch.cat((einops.rearrange(left_coords, 'N xy -> xy N', xy=2),
                         torch.ones((1, left_coords.shape[0]), dtype=left_coords.dtype, device=left_coords.device)),
                        dim=0)
    TRS_right_coords = einops.rearrange(torch.matmul(A, to_proj), 'xy N -> N xy', xy=2)
    TRS_flow = einops.rearrange(TRS_right_coords - left_coords, '(H W) xy -> H W xy',
                                **flow_shape)
    return TRS_flow


def flow2TC(flow, src_coords=None):
    """Convert flow to correspondences.

    args:
        flow: (xy-delta, H, W) array or tensor
        src_coords: [optional] (xy, H*W) source coordinates array or tensor

    returns:
        src_coords: (xy, H*W) source coordinates array or tensor
        dst_coords: (xy, H*W) destination coordinates array or tensor
    """
    flow_flat = einops.rearrange(flow, 'delta H W -> delta (H W)', delta=2)
    flow_shape = einops.parse_shape(flow, 'delta H W')
    if src_coords is None:
        if isinstance(flow_flat, torch.Tensor):
            src_coords = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_flat.device)
        else:
            src_coords = get_featuremap_coords((flow_shape['H'], flow_shape['W']))
    dst_coords = src_coords + flow_flat

    return src_coords, dst_coords


def sample_coords_from_mask(mask, N, replace=False):
    ys, xs = np.nonzero(mask)
    N_mask = len(ys)
    assert replace or (N_mask >= N)
    sampled_ids = np.random.choice(N_mask, size=N, replace=replace)
    samples = np.vstack((xs[sampled_ids], ys[sampled_ids]))
    return samples


def get_H_scaling(H_a2b, pts_in_a):
    """Estimate H scaling factor(s) by computing SVD of affine approximation to the homography
    args:
        H_a2b: (3, 3) homography array
        pts_in_a: (2, N) array of xy points
    returns:
        scales: a list with 2 scales
    """
    pts_in_b = H_proj(H_a2b, pts_in_a)  # (2, N)
    A_a2b, _ = cv2.estimateAffine2D(einops.rearrange(pts_in_a, 'xy N -> N 1 xy', xy=2),
                                    einops.rearrange(pts_in_b, 'xy N -> N 1 xy', xy=2))
    M = A_a2b[:2, :2]
    scales = np.linalg.svd(M, compute_uv=False)
    return scales


def H_lowpass_filter(img, H, contour):
    scales = get_H_scaling(H, contour)
    scale = np.amax(scales)  # take the least downscaling one

    sigma = 1 / scale  # https://dsp.stackexchange.com/a/76015

    eps = 0.1
    if scale < (1 - eps):
        result = cv2.GaussianBlur(img, (0, 0), sigma)
    else:
        result = img
    return result
