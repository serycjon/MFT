# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import numpy as np
import einops
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from PIL import Image, ImageFont, ImageDraw
from ipdb import iex

try:
    from pycolormap_2d import ColorMap2DZiegler
except ImportError:
    pass


import MFT.utils.geom_utils as gu
from MFT.utils.rand_utils import tmp_np_seed
from MFT.utils.interpolation import FlowInterpolator

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def cv2_hatch(canvas, mask, color=(0, 0, 0), alpha=1, **kwargs):
    """ Put a hatching over the canvas, where mask is True """
    hatching = hatch_pattern(canvas.shape[:2], **kwargs)
    hatch_mask = np.logical_and(mask,
                                hatching > 0)
    hatch_overlay = np.einsum("yx,c->yxc", hatch_mask, color).astype(np.uint8)
    alpha = np.expand_dims(hatch_mask * alpha, axis=2)
    vis = alpha * hatch_overlay + (1 - alpha) * canvas
    return vis.astype(np.uint8)


def hatch_pattern(shape, normal=(2, 1), spacing=10, full=False, **kwargs):
    """ Create a parralel line hatch pattern.  Or a fully filled pattern if 'full' is True.

    Args:
        shape - (H, W) canvas size
        normal - (x, y) line normal vector (doesn't have to be normalized)
        spacing - size of gap between the lines in pixels
        full - use solid fill instead of the pattern 

    Outputs:
        canvas - <HxW> np.uint8 image with parallel lines, such that (normal_x, normal_y, c) * (c, r, 1) = 0
    """
    line_type = kwargs.get('line_type', cv2.LINE_8)

    H, W = shape[:2]
    if full:
        canvas = 255 * np.ones((H, W), dtype=np.uint8)
        return canvas

    canvas = np.zeros((H, W), dtype=np.uint8)
    normal = np.array(normal)
    normal = normal / np.sqrt(np.sum(np.square(normal)))

    corners = np.array([[0, 0],
                        [0, H],
                        [W, 0],
                        [W, H]])
    distances = np.einsum("ij,j->i", corners, normal)
    min_c = np.amin(distances)
    max_c = np.amax(distances)
    for c in np.arange(min_c, max_c, spacing):
        res = img_line_pts((H, W), (normal[0], normal[1], -c))
        if not res:
            continue
        else:
            pt_a, pt_b = res
            cv2.line(canvas,
                     tuple(int(x) for x in pt_a),
                     tuple(int(x) for x in pt_b),
                     255,
                     1,
                     line_type)
    return canvas


def img_line_pts(img_shape, line_eq):
    """ Return boundary points of line in image or False if no exist

    Args:
        img_shape - (H, W) tuple
        line_eq   - 3-tuple (a, b, c) such that ax + by + c = 0

    Returns:
        (x1, y1), (x2, y2) - image boundary intersection points
        or False, if the line doesn't intersect the image
    """
    a, b, c = (float(x) for x in line_eq)
    H, W = img_shape
    if a == 0 and b == 0:
        raise ValueError("Invalid line equation: {}".format(line_eq))
    elif a == 0:
        y = -c / b
        if y < 0 or y >= H:
            return False
        else:
            return (0, y), (W, y)

    elif b == 0:
        x = -c / a
        if x < 0 or x >= W:
            return False
        else:
            return (x, 0), (x, H)
    else:
        pts = set([])

        X_y0_intersection = -c / a
        X_yH_intersection = (-c - b * H) / a

        y0_in = X_y0_intersection >= 0 and X_y0_intersection <= W
        yH_in = X_yH_intersection >= 0 and X_yH_intersection <= W
        if y0_in:
            pts.add((X_y0_intersection, 0))
        if yH_in:
            pts.add((X_yH_intersection, H))

        Y_x0_intersection = -c / b
        Y_xW_intersection = (-c - a * W) / b

        x0_in = Y_x0_intersection >= 0 and Y_x0_intersection <= H
        xW_in = Y_xW_intersection >= 0 and Y_xW_intersection <= H
        if x0_in:
            pts.add((0, Y_x0_intersection))
        if xW_in:
            pts.add((W, Y_xW_intersection))

        if len(pts) == 0:
            return False
        elif len(pts) == 1:
            return False
        elif len(pts) == 2:
            return pts.pop(), pts.pop()
        else:
            raise RuntimeError("Found {} intersections! {}".format(len(pts), pts))


def cv2_colorbar(img, vmin, vmax, cmap=None):
    if cmap is None:
        cmap = plt.cm.viridis

    if img.shape[1] < 300:
        scale = int(np.ceil(300 / img.shape[1]))
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cbar_thickness = 20
    separator_sz = 1
    cbar_length = img.shape[1]
    cbar = np.linspace(vmin, vmax, cbar_length, dtype=np.float32)
    cbar = np.tile(cbar, (cbar_thickness, 1))
    cbar = (255 * cmap(norm(cbar))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    separator = np.zeros((separator_sz, cbar.shape[1], cbar.shape[2]), dtype=img.dtype)

    # .copy() to ensure contiguous array? otherwise cv2.putText fails.
    vis = np.vstack((img, separator, cbar)).copy()

    text_margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    thickness = 1

    text_min = '{:.2f}'.format(vmin)
    text_min_size, text_min_baseline = cv2.getTextSize(text_min, font, size, thickness)
    text_min_bl = (text_margin,
                   img.shape[0] - (text_margin + text_min_baseline + separator_sz))
    cv2.putText(vis, text_min,
                text_min_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    text_max = '{:.2f}'.format(vmax)
    text_max_size, text_max_baseline = cv2.getTextSize(text_max, font, size, thickness)
    text_max_bl = (img.shape[1] - (text_margin + text_max_size[0]),
                   img.shape[0] - (text_margin + text_max_baseline + separator_sz))
    cv2.putText(vis, text_max,
                text_max_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis.copy()


def plt_hatch(mask, ax):
    """ https://stackoverflow.com/a/51345660/1705970 """
    ax.contourf(mask, 1, hatches=['', '//'], alpha=0.)


def get_cmap(name):
    return plt.get_cmap(name)


def cv2_cmap_get(cmap, value):
    # RGBA -> BGR
    return (255 * np.array(cmap(value))[..., ::-1][..., 1:]).astype(np.uint8)


def cv2_colormap(img, cmap=None, vmin=None, vmax=None, do_colorbar=False, hatch_params=None):
    """ E.g.: vis = colormap(img, plt.cm.viridis) """
    if cmap is None:
        cmap = plt.cm.viridis
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    vis = (255 * cmap(norm(img))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    vis[np.isnan(img)] = 0

    if hatch_params is not None:
        vis = cv2_hatch(vis, **hatch_params)

    if do_colorbar:
        vis = cv2_colorbar(vis, vmin, vmax, cmap)

    return vis.copy()


def colormap_value(value, vmin, vmax, cmap=None):
    if cmap is None:
        cmap = plt.cm.viridis

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    r, g, b, a = cmap(norm(value))
    return int(255 * b), int(255 * g), int(255 * r)


def to_gray_3ch(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)


def vis_alignment_plain(src, dst, equalize_hist=False):
    assert src.shape == dst.shape

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if equalize_hist:
        dst_gray = cv2.equalizeHist(dst_gray).astype(np.float32) / 255
        src_gray = cv2.equalizeHist(src_gray).astype(np.float32) / 255
    else:
        dst_gray = dst_gray.astype(np.float32) / 255
        src_gray = src_gray.astype(np.float32) / 255
        dst_gray = (dst_gray - np.amin(dst_gray)) / np.ptp(dst_gray)
        src_gray = (src_gray - np.amin(src_gray)) / np.ptp(src_gray)

    alignment_vis = np.zeros(dst.shape, dtype=np.float32)
    alignment_vis[:, :, 0] = dst_gray
    alignment_vis[:, :, 1] = src_gray
    alignment_vis[:, :, 2] = dst_gray

    alignment_vis = np.uint8(alignment_vis * 255)
    return alignment_vis


def vis_alignment(init_img, current_img, H_cur2init, init_xywh,
                  margin=0,
                  show_frames=True):
    """ https://blogs.mathworks.com/steve/2021/01/05/how-imshowpair-and-imfuse-work/
    https://www.mathworks.com/help/images/ref/imfuse.html """
    init_bbox = gu.Bbox.from_xywh(init_xywh)
    crop_bbox = gu.Bbox.from_xyxy((init_bbox.tl_x - int(margin * init_bbox.w),
                                   init_bbox.tl_y - int(margin * init_bbox.h),
                                   init_bbox.br_x + int(margin * init_bbox.w),
                                   init_bbox.br_y + int(margin * init_bbox.h)))
    crop_xywh = crop_bbox.as_xywh()
    template_crop_H = gu.H_bbox2bbox(crop_bbox,
                                     gu.Bbox.from_xywh((0, 0, crop_bbox.w, crop_bbox.h)),)
    in_crop_init_bbox = gu.project_bbox(init_bbox, template_crop_H)
    int_in_crop_init_bbox = in_crop_init_bbox.rounded_to_int()

    current_warped = cv2.warpPerspective(current_img, H_cur2init,
                                         (init_img.shape[1],
                                          init_img.shape[0]))
    template = gu.cv_crop_bbox(init_img, crop_xywh)
    current_in_template = gu.cv_crop_bbox(current_warped, crop_xywh)

    alignment_vis = vis_alignment_plain(current_in_template, template)

    template = draw_box_with_margins(template, int_in_crop_init_bbox, radius=5)
    current_in_template = draw_box_with_margins(current_in_template, int_in_crop_init_bbox, radius=5)
    alignment_vis = draw_box_with_margins(alignment_vis, int_in_crop_init_bbox, radius=5)

    if show_frames:
        alignment_vis = np.concatenate((template, current_in_template, alignment_vis),
                                       axis=1)

    # diff = np.float32(template - current_in_template) / 255
    # dist = np.sqrt(np.sum(np.square(diff), axis=2))
    # dist_vis = vis_utils.cv2_colormap(dist,  # plt.cm.plasma,
    #                                   do_colorbar=False).astype(np.float32)
    # dist_vis = dist_vis.astype(np.uint8)
    # return dist_vis

    return alignment_vis


def draw_box_with_margins(canvas, bbox, radius, color=(0, 0, 255)):
    vis = canvas.copy()
    cv2.circle(vis, (bbox.tl_x, bbox.tl_y),
               radius, color)
    cv2.circle(vis, (bbox.br_x, bbox.tl_y),
               radius, color)
    cv2.circle(vis, (bbox.br_x, bbox.br_y),
               radius, color)
    cv2.circle(vis, (bbox.tl_x, bbox.br_y),
               radius, color)
    return vis


def vis_prosac_scores(sorted_coords, image):
    canvas = to_gray_3ch(image)
    cmap = plt.cm.viridis

    xy, N = sorted_coords.shape
    assert xy == 2

    top_N = int(N * 0.2)

    norm = plt.Normalize(vmin=0, vmax=top_N)
    for i in range(N):
        score = top_N - i - 1
        if i < top_N:
            color = np.uint8(np.array(cmap(norm(score))[2::-1]) * 255).tolist()
        else:
            color = [30, 30, 30]

        x = int(np.round(sorted_coords[0, i]))
        y = int(np.round(sorted_coords[1, i]))

        cv2.circle(canvas, (x, y), 2, color, -1)
    return canvas


def shorten_text(text, max_len, keep='right'):
    if len(text) <= max_len:
        return text

    margin = 3 if max_len > 8 else 0
    if keep == 'right':
        return "." * margin + text[-(max_len - margin):]
    elif keep == 'left':
        return text[:(max_len - margin)] + "." * margin
    else:
        raise ValueError("Invalid 'keep' value")


def draw_text(img, text, size=3, color=(255, 255, 255),
              pos='bl', thickness=3,
              bg=True, bg_alpha=0.5,
              fit_in=True):
    canvas = img.copy()
    text_margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    if isinstance(text, (list, tuple)):
        texts = text
    else:
        texts = [text]

    if isinstance(color, (list, tuple)) and isinstance(color[0], int):
        colors = [color]
        bg_color = (255 - color[0], 255 - color[1], 255 - color[2])
    elif isinstance(color, (list, tuple)) and len(color) == len(texts):
        colors = color
        bg_color = (0, 0, 0)
    else:
        raise ValueError("weird color input")

    while True:
        text_size, text_baseline = cv2.getTextSize(''.join(texts), font, size, thickness)
        text_W, text_H = text_size
        if fit_in:
            if (text_W > canvas.shape[1] or text_H > canvas.shape[0]) and size > 1:
                # try again with smaller size
                size = max(1, size - 1)
                thickness = max(1, thickness - 1)
            else:
                break  # found a fitting size, or cannot shrink any more
        else:
            break
    if pos == 'bl':
        text_bl = (text_margin, canvas.shape[0] - (text_margin + text_baseline))
    elif pos == 'tr':
        text_bl = (canvas.shape[1] - (text_margin + text_W),
                   text_margin + text_H)
    elif pos == 'tl':
        text_bl = (text_margin,
                   text_margin + text_H)
    elif pos == 'br':
        text_bl = (canvas.shape[1] - (text_margin + text_W),
                   canvas.shape[0] - (text_margin + text_baseline))
    elif pos == 'center':
        text_bl = (canvas.shape[1] // 2 - text_W // 2,
                   canvas.shape[0] // 2 + text_H // 2)
    else:
        text_bl = pos

    if bg:
        bg_margin = 2
        bg_canvas = canvas.copy()
        cv2.rectangle(bg_canvas,
                      (text_bl[0] - bg_margin, text_bl[1] + bg_margin + text_baseline),
                      (text_bl[0] + text_W + bg_margin, text_bl[1] - text_H - bg_margin),
                      bg_color, thickness=-1)
        canvas = cv2.addWeighted(bg_canvas, bg_alpha, canvas, (1 - bg_alpha), 0)

    bl = text_bl
    for text_i, text in enumerate(texts):
        c = colors[text_i % len(colors)]
        canvas = cv2.putText(canvas, text,
                             bl, font,
                             size, c, thickness, cv2.LINE_AA)
        bl = (bl[0] + cv2.getTextSize(text, font, size, thickness)[0][0], bl[1])
    return canvas


def draw_text_multiline_centered(img, texts, center_x, center_y,
                                 size=3, color=(255, 255, 255), thickness=3, font=cv2.FONT_HERSHEY_SIMPLEX):
    canvas = img.copy()

    text_sizes = []
    for text in texts:
        text_size, text_baseline = cv2.getTextSize(text, font, size, thickness)
        # size is a tuple (width, height)
        # baseline is y-coordinate relative to bottom-most text point
        text_sizes.append((text_size, text_baseline))

    max_width = max(size[0][0] for size in text_sizes)
    x_shifts = [(max_width - text_size[0]) // 2 for text_size, _ in text_sizes]

    y_gap = max(size[0][1] for size in text_sizes) // 2
    total_height = sum(size[0][1] for size in text_sizes) + y_gap * (len(text_sizes) - 1)

    y_shift = 0
    for text, x_shift, text_size in reversed(list(zip(texts, x_shifts, text_sizes))):
        bottom_left = (center_x - max_width // 2 + x_shift,
                       center_y + total_height // 2 - y_shift)
        y_shift += text_size[0][1] + y_gap
        canvas = cv2.putText(canvas, text, bottom_left, font, size, color, thickness, cv2.LINE_AA)
    return canvas


def draw_text_multiline_centered_PIL(img, texts, center_x, center_y,
                                     size=60, color=(255, 255, 255), thickness=1, fontname='DejaVuSans'):
    img_pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(fontname, size)
    extra_thickness = 8
    draw.text((center_x, center_y), '\n'.join(texts), (0, 0, 0), font=font, anchor='mm', align='center',
              stroke_width=thickness + extra_thickness, spacing=4)
    draw.text((center_x, center_y), '\n'.join(texts), color, font=font, anchor='mm', align='center',
              stroke_width=thickness, spacing=4 + 2 * extra_thickness)

    img_np = np.asarray(img_pil)[:, :, ::-1]
    return img_np


def draw_corners(canvas, corners, color, thickness=2, with_cross=True, with_TL=False):
    """ Draw polygon bounded by corners

    args:
        corners: (2, 4) array
    """
    if corners is None:
        return canvas
    assert corners.shape == (2, 4), f"Incorrect corners shape {corners.shape}"
    # pts = np.round(corners).astype(np.int32)
    pts = corners
    pts = einops.rearrange(pts, 'xy N_corners -> N_corners 1 xy', xy=2, N_corners=4)
    vis = canvas.copy()
    vis = polylines(vis, [pts], True, color, thickness)
    if with_cross:
        vis = line(vis, tuple(pts[0, 0, :]), tuple(pts[2, 0, :]), color, thickness)
        vis = line(vis, tuple(pts[1, 0, :]), tuple(pts[3, 0, :]), color, thickness)
    if with_TL:
        vis = cv2.circle(vis, tuple(pts[0, 0, :].astype(np.int32).tolist()), radius=2 * thickness, color=color, thickness=-1)
    return vis


def vis_vector_in_center(img, vector, color=(0, 255, 0), thickness=2, shift=4):
    canvas = img.copy()
    center = canvas.shape[1] // 2, canvas.shape[0] // 2
    line(canvas, center, (center[0] + vector[0],
                          center[1] + vector[1]),
         color, thickness, shift=shift)
    return canvas


def imshow(win_name, img, scalable=True):
    flags = cv2.WINDOW_AUTOSIZE
    if scalable:
        flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(win_name, flags)
    cv2.imshow(win_name, img)


def line(img, pt1, pt2, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.line, but does the fractional bit shift inside
    accepts float pt1 and pt2."""
    if shift is not None:
        multiplier = 2**shift
        pt1 = tuple(np.round(multiplier * np.array(pt1)).astype(np.int32).tolist())
        pt2 = tuple(np.round(multiplier * np.array(pt2)).astype(np.int32).tolist())

    return cv2.line(img, pt1, pt2, color, thickness, lineType, shift)


def circle(img, center, radius, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.circle, but does the fractional bit shift inside
    accepts float center and radius."""
    if shift is not None:
        multiplier = 2**shift
        center = tuple(np.round(multiplier * np.array(center)).astype(np.int32).tolist())
        radius = int(np.round(multiplier * radius).astype(np.int32))

    return cv2.circle(img, center, radius, color, thickness, lineType, shift)


def polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.polylines, but does the fractional bit shift inside
    accepts float pts."""
    if shift is not None:
        multiplier = 2**shift
        pts = np.round(multiplier * np.array(pts)).astype(np.int32)
    return cv2.polylines(img, pts, isClosed, color, thickness, lineType, shift=shift)


def place_img_at(img, canvas, tl_row, tl_col):
    H, W = img.shape[:2]
    canvas[tl_row:tl_row + H, tl_col:tl_col + W, :] = img


def name_fig(img_list, name_list, size=1, thickness=1, pos='tl'):
    named = []
    for img, name in zip(img_list, name_list):
        named.append(draw_text(img, name, size=size, thickness=thickness, pos=pos))
    return named


def griddify(img_list, cols=None, rows=None):
    N_img = len(img_list)
    if cols is None and rows is None:
        cols = int(np.floor(np.sqrt(N_img)))

    if cols is None:
        cols = int(np.ceil(N_img / rows))

    if rows is None:
        rows = int(np.ceil(N_img / cols))

    grid = []
    for row in range(rows):
        row_imgs = []
        for col in range(cols):
            i = row * cols + col
            img = None
            if i < N_img:
                img = img_list[i]

            row_imgs.append(img)
        grid.append(row_imgs)
    return grid


def tile(img_grid, h_space=1, w_space=None, bg_color=None):
    """Tile images

    args:
        img_grid: a 2D array of images
    """
    if w_space is None:
        w_space = h_space

    rows = len(img_grid)
    cols = len(img_grid[0])

    row_heights = [0] * rows
    col_widths = [0] * cols

    for row_i, row in enumerate(img_grid):
        for col_i, img in enumerate(row):
            if img is None:
                continue
            H, W = img.shape[:2]
            row_heights[row_i] = max(row_heights[row_i], H)
            col_widths[col_i] = max(col_widths[col_i], W)

    out_H = np.sum(row_heights) + (rows - 1) * h_space
    out_W = np.sum(col_widths) + (cols - 1) * w_space
    canvas = np.zeros((out_H, out_W, 3), dtype=img_grid[0][0].dtype)
    if bg_color is not None:
        canvas[:, :] = bg_color

    cur_row = 0
    for row_i, row in enumerate(img_grid):
        cur_col = 0
        for col_i, img in enumerate(row):
            if img is None:
                continue
            place_img_at(img, canvas, cur_row, cur_col)
            cur_col += col_widths[col_i] + w_space

        cur_row += row_heights[row_i] + h_space

    return canvas


class VideoWriter():
    def __init__(self, path, fps=30, images_export=False, ext='jpg'):
        self.do_write = path is not None
        if self.do_write:
            if images_export:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.path = path
        self.fps = fps
        self.images_export = images_export
        self.frame_i = 0
        self.ext = ext

    def write(self, frame, frame_name=None):
        if self.writer is None and self.do_write and not self.images_export:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                          (frame.shape[1], frame.shape[0]))

        if self.do_write:
            if self.images_export:
                if frame_name is not None:
                    out_path = self.path / f'{frame_name}.{self.ext}'
                else:
                    out_path = self.path / f'{self.frame_i:08d}.{self.ext}'

                cv2.imwrite(str(out_path), frame)
                self.frame_i += 1
            else:
                self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()

    def __del__(self):
        self.close()


def make_knn_interp(db_xy, db_values):
    """ Create a K-NN interpolator

    args:
        db_xy:      (N, D) array (N samples of dimension D)
        db_values:  (N, ) array of values corresponding to the xy positions

    returns:
        interpolation function
    """
    tree = KDTree(db_xy)
    N = db_xy.shape[0]

    def interp(query_xy, K, max_dist=None, min_K=None):
        """ interpolate query data using K-NN.  Invalid data represented by np.nan

        args:
            query_xy: (..., D) array of D-dimensional queries
            K: number of interpolated nearest neighbors
            max_dist: maximum distance for K-NN search
            min_K: minimum number of NNs found for interpolation to happen
        """
        distance_bound = max_dist
        if distance_bound is None:
            distance_bound = np.inf

        dists, ids = tree.query(query_xy, k=K, distance_upper_bound=distance_bound)

        valid_mask = ids < N
        dists[np.logical_not(valid_mask)] = np.nan

        # weights = np.exp(-dists) / np.nansum(np.exp(-dists), axis=-1, keepdims=True)
        weights = (-dists) / np.nansum(-dists, axis=-1, keepdims=True)
        # weights = (dists) / np.nansum(dists, axis=-1, keepdims=True)

        vals = np.full(dists.shape, np.nan)
        vals[valid_mask] = db_values[ids[valid_mask]]
        # vals[valid_mask] *= np.exp(dists[valid_mask]) / np.sum(np.exp(dists[valid_mask], axis=-1))
        vals[valid_mask] *= weights[valid_mask]
        vals = np.nansum(vals, axis=-1)
        invalid = np.logical_not(np.any(valid_mask, axis=-1))
        if min_K is not None:
            K_found = np.sum(valid_mask, axis=-1)
            invalid = np.logical_or(invalid, K_found < min_K)
        vals[invalid] = np.nan
        return vals

    return interp


def blend_mask(img, mask, color=(0, 255, 0), alpha=0.5,
               fill=True, contours=True, contour_thickness=1,
               confidence=None):
    '''Blend color mask over image

    img -- 3 channel float32 img with 0-255 values
    mask -- numpy single channel bool img
    color -- three-tuple with 0-255 values RGB
    alpha -- float mask alpha
    contours -- whether to draw mask contours (with alpha=1)
    contour_thickness -- pixel thickness of the contours

    Adapted from:
    https://github.com/karolmajek/Mask_RCNN/blob/master/visualize.py

    '''
    canvas = img.copy()
    if confidence is not None:
        alpha = alpha * confidence

    if fill:
        color_array = np.array(color)[np.newaxis, np.newaxis, :]
        canvas[mask, :] = canvas[mask] * (1 - alpha) + alpha * color_array

    if contours:
        cnt = compatible_contours(mask.astype(np.uint8) * 255, retrieval_mode=cv2.RETR_LIST)
        cv2.drawContours(canvas, cnt, -1, color, contour_thickness)

    return canvas


def blend_with_alpha(img1, img2, img1_alpha):
    """Blend two uint8 images using an alpha channel"""
    if img1_alpha.max() > 1.0001:
        img1_alpha = img1_alpha.astype(np.float32) / 255.0

    if len(img1_alpha.shape) == 2:
        img1_alpha = einops.rearrange(img1_alpha, 'H W -> H W 1')

    result = img1.astype(np.float32) * img1_alpha + img2.astype(np.float32) * (1 - img1_alpha)
    result = result.clip(0, 255).astype(np.uint8)
    return result


def blend_with_alpha_premult(img1_premult, img2, img1_alpha):
    """Blend two uint8 images using an alpha channel"""
    if img1_alpha.max() > 1.0001:
        img1_alpha = img1_alpha.astype(np.float32) / 255.0

    if len(img1_alpha.shape) == 2:
        img1_alpha = einops.rearrange(img1_alpha, 'H W -> H W 1')

    result = img1_premult.astype(np.float32) + img2.astype(np.float32) * (1 - img1_alpha)
    result = result.clip(0, 255).astype(np.uint8)
    return result


def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")


def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")


def is_cv4():
    # if we are using OpenCV 4.X, then our cv2.__version__ will start
    # with '4.'
    return check_opencv_version("4.")


def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)


def compatible_contours(thresh, retrieval_mode=cv2.RETR_EXTERNAL):
    if is_cv2():
        (contours, _) = cv2.findContours(thresh, retrieval_mode,
                                         cv2.CHAIN_APPROX_SIMPLE)
    elif is_cv3():
        (_, contours, _) = cv2.findContours(thresh, retrieval_mode,
                                            cv2.CHAIN_APPROX_SIMPLE)
    elif is_cv4():
        contours, _ = cv2.findContours(thresh, retrieval_mode,
                                       cv2.CHAIN_APPROX_SIMPLE)
    else:
        print('cv2.__version__: {}'.format(cv2.__version__))
        raise RuntimeError("unknown opencv version!")
    return contours


def plt_to_img(fig, close=True):
    """https://stackoverflow.com/a/57988387/1705970"""
    # ax.axis('off')
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    # ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if close:
        plt.close()

    return image_from_plot[:, :, ::-1]


def cv_plt_show(fig=None, close=True):
    if fig is None:
        fig = plt.gcf()
    img = plt_to_img(fig, close)
    cv2.imshow("cv: img", img)
    while True:
        c = cv2.waitKey(0)
        if c == ord('q'):
            break


def vis_flow_align_new(flow, src_img, dst_img, *args, **kwargs):
    vis = vis_alignment_plain(src_img, dst_img)
    return vis_flow(flow, vis, vis, *args, **kwargs)

def vis_flow_dots(flow, src_img, dst_img, grid_sz=30,
                  dot_radius=3, cmap='viridis',
                  mask=None):
    H, W = flow.shape[:2]
    assert flow.shape[2] in [2, 3]

    src_xs = np.arange(0, W, grid_sz)
    src_ys = np.arange(0, H, grid_sz)

    xs, ys = np.meshgrid(src_xs, src_ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    pts_dst = flow[flat_ys, flat_xs, :2]
    pts_dst[:, 0] += flat_xs
    pts_dst[:, 1] += flat_ys

    pts_src = np.vstack((flat_xs, flat_ys))
    pts_dst = pts_dst.T

    if mask is not None:
        pt_mask = mask[pts_src[1, :], pts_src[0, :]]
        pts_src = pts_src[:, pt_mask]
        pts_dst = pts_dst[:, pt_mask]

    src_vis = src_img.copy()
    dst_vis = dst_img.copy()

    cmap = cm.get_cmap(cmap)
    norm = plt.Normalize(np.amin(pts_src[0, :]), np.amax(pts_src[0, :]))

    for i in range(pts_src.shape[1]):
        a = pts_src[:, i]
        b = pts_dst[:, i]

        color = (255 * np.array(cmap(norm(a[0]))))[[2, 1, 0]].astype(np.int32).tolist()

        circle(src_vis, a, radius=dot_radius,
               thickness=-1, color=color)
        circle(dst_vis, b, radius=dot_radius,
               thickness=-1, color=color)
    return src_vis, dst_vis

def vis_flow(flow, src_img, dst_img, grid_sz=10,
             occl=None, occl_thr=255,
             arrow_color=(0, 0, 255),
             point_color=(0, 255, 255),
             point_radius=0,
             occlusion_color=None,
             vis_alpha=1,
             decimal_places=2):
    """ Flow field visualization

    Args:
        flow             - <H x W x 3> flow with channels (u, v, _)
                            (u - flow in x direction, v - flow in y direction)
        src_img          - <H x W x 3> BGR flow source image
        dst_img          - <H x W x 3> BGR flow destination image
        grid_sz          - visualization grid size in pixels
        occl             - <H x W> np.uint8 soft occlusion mask (0-255)
        occl_thr         - (0-255) occlusion threshold (occl >= occl_thr means occlusion)
        arrow_color      - BGR 3-tuple of flow arrow color
        point_color      - BGR 3-tuple of flow point color
        occlusion_color  - None, or BGR 3-tuple of flow point color (not drawn if None)
        decimal_places   - number of decimal places to be used for positions

    Returns:
        src_vis - <H x W x 3> BGR flow visualization in source image
        dst_vis - <H x W x 3> BGR flow visualization in destination image
    """
    line_type = cv2.LINE_AA  # cv2.LINE_8 or cv2.LINE_AA - antialiased, but blurry...
    circle_type = cv2.LINE_AA

    shift = int(np.ceil(np.log2(10**decimal_places)))
    pt_radius = point_radius * (2**shift)

    H, W = flow.shape[:2]
    assert flow.shape[2] in [2, 3]

    src_xs = np.arange(W)
    src_ys = np.arange(H)

    xs, ys = np.meshgrid(src_xs, src_ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    pts_dst = flow[flat_ys, flat_xs, :2]
    pts_dst[:, 0] += flat_xs
    pts_dst[:, 1] += flat_ys

    pts_src = np.vstack((flat_xs, flat_ys))
    pts_dst = pts_dst.T

    if True:
        mask = np.all(np.mod(pts_src, grid_sz) == 0, axis=0)
    # else:
    #     mask = np.zeros(pts_src.shape[1])
    #     from scipy.stats import qmc
    #     rng = np.random.default_rng()
    #     engine = qmc.PoissonDisk(d=2, radius=0.01, seed=rng)
    #     sample = engine.fill_space()
    #     sample[:, 0] *= H
    #     sample[:, 1] *= W
    #     sample = np.round(sample).astype(np.int32)

    pts_src = np.round(pts_src * (2**shift)).astype(np.int32)
    pts_dst = np.round(pts_dst * (2**shift)).astype(np.int32)

    src_vis = src_img.copy()
    dst_vis = dst_img.copy()

    backgrounds = OverlayBackground(src_vis, dst_vis)
    # draw flow lines/arrows
    if arrow_color == 'rand':
        with tmp_np_seed(42):
            rand_colors = np.random.randint(0, 255, size=(H * W, 3)).tolist()
    for i in range(mask.size):
        if arrow_color == 'rand':
            color = tuple(rand_colors[i])
        else:
            color = arrow_color
        if mask[i]:
            if (occl is not None) and (occlusion_color is None) and (occl[np.unravel_index(i, occl.shape)] >= occl_thr):
                continue
            a = pts_src[:, i]
            b = pts_dst[:, i]

            cv2.line(src_vis,
                     (a[0], a[1]),
                     (b[0], b[1]),
                     color,
                     lineType=line_type,
                     shift=shift)
            cv2.line(dst_vis,
                     (a[0], a[1]),
                     (b[0], b[1]),
                     color,
                     lineType=line_type,
                     shift=shift)

    # draw flow points
    for i in range(mask.size):
        if mask[i]:
            a = pts_src[:, i]
            b = pts_dst[:, i]

            if occl is not None and occl[np.unravel_index(i, occl.shape)] >= occl_thr:
                occluded = True
            else:
                occluded = False

            if occluded and occlusion_color is None:
                continue

            cv2.circle(src_vis,
                       (a[0], a[1]),
                       radius=pt_radius,
                       color=point_color,
                       lineType=circle_type,
                       shift=shift)
            cv2.circle(dst_vis,
                       (b[0], b[1]),
                       radius=pt_radius,
                       color=point_color if not occluded else occlusion_color,
                       lineType=circle_type,
                       shift=shift)

    src_vis, dst_vis = backgrounds.overlay(src_vis, dst_vis, alpha=vis_alpha)

    return src_vis, dst_vis


def vis_flow_align(flow, src_img, dst_img, grid_sz=10,
                   arrow_color=(0, 0, 255),
                   decimal_places=2,
                   show_flow=True,
                   show_start=True,
                   show_end=False,
                   pt_radius=0):
    """ Flow field visualization

    Args:
        flow             - <H x W x 3> flow with channels (u, v, _)
                            (u - flow in x direction, v - flow in y direction)
        src_img          - <H x W x 3> BGR flow source image
        dst_img          - <H x W x 3> BGR flow destination image
        grid_sz          - visualization grid size in pixels
        arrow_color      - BGR 3-tuple of flow arrow color
        point_color      - BGR 3-tuple of flow point color
        decimal_places   - number of decimal places to be used for positions

    Returns:
        src_vis - <H x W x 3> BGR flow visualization in source image
        dst_vis - <H x W x 3> BGR flow visualization in destination image
    """
    line_type = cv2.LINE_AA  # cv2.LINE_8 or cv2.LINE_AA - antialiased, but blurry...
    circle_type = cv2.LINE_8

    shift = int(np.ceil(np.log2(10**decimal_places)))

    H, W = flow.shape[:2]

    src_xs = np.arange(W)
    src_ys = np.arange(H)

    xs, ys = np.meshgrid(src_xs, src_ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    pts_dst = flow[flat_ys, flat_xs, :2]
    pts_dst[:, 0] += flat_xs
    pts_dst[:, 1] += flat_ys

    pts_src = np.vstack((flat_xs, flat_ys))
    pts_dst = pts_dst.T
    N_pts = pts_src.shape[1]

    mask = np.all(np.mod(pts_src, grid_sz) == 0, axis=0)

    pts_src = np.round(pts_src * (2**shift)).astype(np.int32)
    pts_dst = np.round(pts_dst * (2**shift)).astype(np.int32)
    pt_radius = np.round(pt_radius * (2**shift)).astype(np.int32)

    vis = vis_alignment_plain(src_img, dst_img)

    # draw flow lines/arrows
    for i in range(mask.size):
        if mask[i]:
            a = pts_src[:, i]
            b = pts_dst[:, i]

            alpha = 1
            if len(arrow_color) == 3:  # BGR
                color = arrow_color
            elif len(arrow_color) == 4:  # BGRA
                color = tuple(arrow_color[:3])
                alpha = arrow_color[3]
            elif arrow_color.shape == (N_pts, 3):
                color = tuple(arrow_color[i, :].tolist())
            elif arrow_color.shape == (N_pts, 4):
                color = tuple(arrow_color[i, :3].tolist())
                alpha = arrow_color[i, 3]

            if alpha > 0:
                if show_flow:
                    cv2.line(vis,
                             (a[0], a[1]),
                             (b[0], b[1]),
                             color,
                             lineType=line_type,
                             shift=shift)

                if show_start:
                    cv2.circle(vis,
                               (a[0], a[1]),
                               radius=pt_radius,
                               thickness=-1,
                               color=color,
                               # lineType=circle_type,
                               shift=shift)
                if show_end:
                    cv2.circle(vis,
                               (b[0], b[1]),
                               radius=pt_radius,
                               thickness=-1,
                               color=(255, 0, 255),
                               lineType=circle_type,
                               shift=shift)

    return vis


def checkerboard(h, w, c0, c1, blocksize):
    """ https://stackoverflow.com/a/6179649/1705970 """
    tile = einops.repeat(np.array([[c0, c1], [c1, c0]]), 'mini_H mini_W ... -> (mini_H block_H) (mini_W block_W) ...',
                         block_H=blocksize, block_W=blocksize)
    rows = int(np.ceil(h / tile.shape[0]))
    cols = int(np.ceil(w / tile.shape[1]))
    grid = einops.repeat(tile, 'tile_H tile_W ... -> (rows tile_H) (cols tile_W) ...',
                         rows=rows, cols=cols)
    grid = grid[:h, :w, ...]

    return grid


def color_checkerboard(h, w, blocksize):
    bool_checkerboard = checkerboard(h, w, 0, 1, blocksize)
    cmap = ColorMap2DZiegler(range_x=(0, w-1), range_y=(0, h-1))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    assert xs.shape == (h, w)
    assert ys.shape == (h, w)
    assert xs[0, 0] == xs[1, 0]
    coords = np.dstack((xs, ys))

    def coloring_fun(xy):
        x, y = xy
        return np.array(cmap(x, y))

    colored = np.apply_along_axis(coloring_fun, axis=2, arr=coords)
    colored[bool_checkerboard == 0, :] = 0
    colored = np.concatenate((colored, 255 * einops.rearrange(bool_checkerboard, 'H W -> H W 1')), axis=2)
    return colored


def color_dotboard(h, w, skip, radius):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cmap = ColorMap2DZiegler(range_x=(0, w-1), range_y=(0, h-1))

    def coloring_fun(xy):
        x, y = xy
        return np.array(cmap(x, y))

    xs = np.arange(0, w-1, skip)
    ys = np.arange(0, h-1, skip)
    for x in xs:
        for y in ys:
            color = cmap(x, y).tolist()
            circle(canvas, (x, y), radius, color, thickness=-1)

    return canvas


class OverlayBackground(object):
    """It would be so nice to have Lisp macros and not to have to deal with Python weakness...

    It would be so nice!

    (with-overlay (img_a img_b :alpha 0.5)
      ... operations on img_a ...
      ... operations on img_b ...
    )

    img_a would now be a 1:1 mix between original img_a and img_a after the operations, same for img_b
    """

    def __init__(self, *background_images):
        self.backgrounds_copy = [img.copy() for img in background_images]

    def overlay(self, *overlay_images, alpha=0.5):
        return [cv2.addWeighted(overlay, alpha, background, (1 - alpha), 0)
                for overlay, background in zip(overlay_images, self.backgrounds_copy)]


class FlowGUI(object):
    def __init__(self, left_img, right_img, flow_left_to_right, mask=None):
        self.left_img = left_img.copy()
        self.right_img = right_img.copy()
        self.mask = mask

        self.left_img_gray = to_gray_3ch(left_img)
        self.right_img_gray = to_gray_3ch(right_img)

        if flow_left_to_right.shape[0] == 2:
            flow_left_to_right = einops.rearrange(flow_left_to_right, 'xy H W -> H W xy', xy=2)
        self.flow_left_to_right = flow_left_to_right
        self.interpolator = FlowInterpolator(flow_left_to_right, mask)

        self.gray = True

    def current_images(self):
        if self.gray:
            return self.left_img_gray.copy(), self.right_img_gray.copy()
        else:
            return self.left_img.copy(), self.right_img.copy()

    def draw(self):
        cv2.namedWindow("cv: left", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cv: left", 800, 600)
        cv2.setMouseCallback("cv: left", self.handler)

        cv2.namedWindow("cv: right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cv: right", 800, 600)

        left_img, right_img = self.current_images()
        cv2.imshow("cv: left", left_img)
        cv2.imshow("cv: right", right_img)

        while True:
            c = cv2.waitKey(0)
            if c == ord('q'):
                break
            elif c == ord('x'):
                sys.exit(1)
            elif c == ord('g'):
                self.gray = not self.gray
            elif c == ord('v'):
                left_img, right_img = self.current_images()
                left_vis, right_vis = vis_flow_align_new(self.flow_left_to_right, left_img, right_img,
                                                         grid_sz=30, occl=np.logical_not(self.mask), occl_thr=0.5)
                cv2.imshow("cv: left_flow", left_vis)

    @iex
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            right_coords = self.interpolator(np.array([[x, y]]))[0, :]
            right_x = x + right_coords[0]
            right_y = y + right_coords[1]
            if len(right_coords) == 3:
                visible = right_coords[2] > 0
            else:
                visible = True

            left_vis, right_vis = self.current_images()
            color = (0, 0, 255)
            if not visible:
                color = (135, 0, 120)
            radius = 6
            left_vis = cv2.circle(left_vis, (x, y), radius=radius, color=color, thickness=-1)
            right_vis = circle(right_vis, (right_x, right_y), radius=radius, color=color, thickness=-1)
            cv2.imshow("cv: left", left_vis)
            cv2.imshow("cv: right", right_vis)
            cv2.waitKey(1)


def zoomins(img, center_coord, zooms, win_size, cross=True, cross_color=(0, 0, 255)):
    results = []
    for zoom in zooms:
        sample_bbox = gu.Bbox.from_cxcywh((center_coord[0], center_coord[1], win_size / zoom, win_size / zoom))
        out_bbox = gu.Bbox.from_xywh((0, 0, win_size, win_size))

        zoomin = sample_bbox.sample_img(img, out_bbox)
        if cross:
            cross_coord = (win_size - 1) / 2
            line(zoomin, (0, cross_coord), (win_size - 1, cross_coord), cross_color, thickness=1)
            line(zoomin, (cross_coord, 0), (cross_coord, win_size - 1), cross_color, thickness=1)
        results.append(zoomin)
    return tile([results])


def resize(src, dsize, fx=None, fy=None, interpolation=None, align_corners=False):
    """resize image with or without align_corners.

    test with:
    img = np.array([[0,1,2,3,4,5]], dtype='float32')
    resize(img, None, 2, 1, align_corners=False)
      -> [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.0]
    resize(img, None, 2, 1, align_corners=True)
      -> [0.0, 0.46875, 0.90625, 1.375, 1.8125, 2.28125, 2.71875, 3.1875, 3.625, 4.09375, 4.53125, 5.0]

    (see https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/)
    """
    if not align_corners:
        return cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)

    if fy is None:
        fy = fx

    if dsize is None or dsize == (0, 0):
        dsize = (int(np.round(fx * src.shape[1])), int(np.round(fy * src.shape[0])))

    if src.shape[1] > 1:
        fx = (dsize[0] - 1) / (src.shape[1] - 1)
    if src.shape[0] > 1:
        fy = (dsize[1] - 1) / (src.shape[0] - 1)
    A = np.array([[fx, 0, 0],
                  [0, fy, 0]], dtype=np.float32)
    return cv2.warpAffine(src, A, dsize, flags=interpolation)
