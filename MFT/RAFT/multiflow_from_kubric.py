"""Kubric dataset with point tracking."""

import functools
import itertools

import matplotlib.pyplot as plt
# import mediapy as media
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
# from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
import cv2
import einops
import pdb
import sys, os
from typing import List, Dict, Union
import imageio
import png
import dataclasses
import json
import time

DEFAULT_LAYERS = ("rgba", "segmentation", "forward_flow", "backward_flow",
                  "depth", "normal", "object_coordinates")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pathlib import Path

# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)

# flow is encoded with sign, so 2**15, occlusion and uncertainty without sign, so 2**16:
FLOWOU_IO_FLOW_MULTIPLIER = 2 ** 5  # max-abs-val = 2**(15-5) = 1024, step = 2**(-5) = 0.03
FLOWOU_IO_OCCLUSION_MULTIPLIER = 2 ** 15  # max-val = 2**(16-15) = 2, step = 2**(-15) = 3e-5
FLOWOU_IO_UNCERTAINTY_MULTIPLIER = 2 ** 9  # max-val = 2**(16-9) = 128, step = 2**(-9) = 0.0019


def from_quaternion(
        quaternion,
        name="rotation_matrix_3d_from_quaternion"):
    """Convert a quaternion to a rotation matrix.

       Note:
         In the following, A1 to An are optional batch dimensions.

       Args:
         quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
           represents a normalized quaternion.
         name: A name for this op that defaults to
           "rotation_matrix_3d_from_quaternion".

       Returns:
         A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
         represent a 3d rotation matrix.

       Raises:
         ValueError: If the shape of `quaternion` is not supported.
    """
    with tf.name_scope(name):
        quaternion = tf.convert_to_tensor(value=quaternion)

        # shape.check_static(
        #     tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
        # quaternion = asserts.assert_normalized(quaternion)

        x, y, z, w = tf.unstack(quaternion, axis=-1)
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                           txy + twz, 1.0 - (txx + tzz), tyz - twx,
                           txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                          axis=-1)  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)


def write_flowou_png(path, flow, occlusions, uncertainty):
    """Write a compressed png flow, occlusions and uncertainty

    Args:
        path: write path (must have ".flowou.png" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible), clipped between 0 and 1
        uncertainty: (1, H, W) array with uncertainty sigma, clipped between 0 and 2047
                      (0 = dirac, max observed on Sintel = 215, Q0.999 on sintel ~ 15)
    """

    def encode_central(xs, multiplier=32.0):
        max_val = 2 ** 15 / multiplier
        assert np.all(np.abs(xs) < max_val), "out-of-range values - cannot be written"
        return 2 ** 15 + multiplier * xs

    def encode_positive(xs, multiplier=32.0):
        max_val = 2 ** 16 / multiplier
        assert np.all(xs >= 0), "out-of-range values - cannot be written"
        assert np.all(xs < max_val), "out-of-range values - cannot be written"
        return multiplier * xs

    assert Path(path).suffixes == ['.flowou', '.png']
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


def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def mkdir_from_full_file_path_if_not_exist(path):
    basename = os.path.basename(path)
    mkdir_if_not_exist(path[:-len(basename)])


# @tf.autograph.experimental.do_not_convert
def project_point(cam, point3d, num_frames):
    """Compute the image space coordinates [0, 1] for a set of points.

    Args:
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      point3d: Points in 3D world coordinates.  it has shape [num_frames,
        num_points, 3].
      num_frames: The number of frames in the video.

    Returns:
      Image coordinates in 2D.  The last coordinate is an indicator of whether
        the point is behind the camera.
    """

    homo_transform = tf.linalg.inv(cam['matrix_world'])
    homo_intrinsics = tf.zeros((num_frames, 3, 1), dtype=tf.float32)
    homo_intrinsics = tf.concat([cam['intrinsics'], homo_intrinsics], axis=2)

    point4d = tf.concat([point3d, tf.ones_like(point3d[:, :, 0:1])], axis=2)
    projected = tf.matmul(point4d, tf.transpose(homo_transform, (0, 2, 1)))
    projected = tf.matmul(projected, tf.transpose(homo_intrinsics, (0, 2, 1)))
    image_coords = projected / projected[:, :, 2:3]
    image_coords = tf.concat(
        [image_coords[:, :, :2],
         tf.sign(projected[:, :, 2:])], axis=2)
    return image_coords


# @tf.autograph.experimental.do_not_convert
def unproject(coord, cam, depth):
    """Unproject points.

    Args:
      coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
        integer (y,x) because of the way meshgrid happens.
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      depth: Depth map for the scene.

    Returns:
      Image coordinates in 3D.
    """
    shp = tf.convert_to_tensor(tf.shape(depth))
    idx = coord[:, 0] * shp[1] + coord[:, 1]
    coord = tf.cast(coord[..., ::-1], tf.float32)
    shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]

    # Need to convert from pixel to raster coordinate.
    projected_pt = (coord + 0.5) / shp

    projected_pt = tf.concat(
        [
            projected_pt,
            tf.ones_like(projected_pt[:, -1:]),
        ],
        axis=-1,
    )

    camera_plane = projected_pt @ tf.linalg.inv(tf.transpose(cam['intrinsics']))
    camera_ball = camera_plane / tf.sqrt(
        tf.reduce_sum(
            tf.square(camera_plane),
            axis=1,
            keepdims=True,
        ), )
    camera_ball *= tf.gather(tf.reshape(depth, [-1]), idx)[:, tf.newaxis]

    camera_ball = tf.concat(
        [
            camera_ball,
            tf.ones_like(camera_plane[:, 2:]),
        ],
        axis=1,
    )
    points_3d = camera_ball @ tf.transpose(cam['matrix_world'])
    return points_3d[:, :3] / points_3d[:, 3:]


# @tf.autograph.experimental.do_not_convert
def reproject(coords, camera, camera_pos, num_frames, bbox=None):
    """Reconstruct points in 3D and reproject them to pixels.

    Args:
      coords: Points in 3D.  It has shape [num_points, 3].  If bbox is specified,
        these are assumed to be in local box coordinates (as specified by kubric),
        and bbox will be used to put them into world coordinates; otherwise they
        are assumed to be in world coordinates.
      camera: the camera intrinsic parameters, as returned by kubric.
        'matrix_world' and 'intrinsics' have a leading axis num_frames.
      camera_pos: the camera positions.  It has shape [num_frames, 3]
      num_frames: the number of frames in the video.
      bbox: The kubric bounding box for the object.  Its first axis is num_frames.

    Returns:
      Image coordinates in 2D and their respective depths.  For the points,
      the last coordinate is an indicator of whether the point is behind the
      camera.  They are of shape [num_points, num_frames, 3] and
      [num_points, num_frames] respectively.
    """
    # First, reconstruct points in the local object coordinate system.

    if bbox is not None:
        coord_box = list(itertools.product([-.5, .5], [-.5, .5], [-.5, .5]))
        coord_box = np.array([np.array(x) for x in coord_box])
        coord_box = np.concatenate(
            [coord_box, np.ones_like(coord_box[:, 0:1])], axis=1)
        coord_box = tf.tile(coord_box[tf.newaxis, ...], [num_frames, 1, 1])
        bbox_homo = tf.concat([bbox, tf.ones_like(bbox[:, :, 0:1])], axis=2)

        local_to_world = tf.linalg.lstsq(tf.cast(coord_box, tf.float32), bbox_homo)
        world_coords = tf.matmul(
            tf.cast(
                tf.concat([coords, tf.ones_like(coords[:, 0:1])], axis=1),
                tf.float32)[tf.newaxis, :, :], local_to_world)
        world_coords = world_coords[:, :, 0:3] / world_coords[:, :, 3:]
    else:
        world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

    # world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

    # Compute depths by taking the distance between the points and the camera
    # center.
    depths = tf.sqrt(
        tf.reduce_sum(
            tf.square(world_coords - camera_pos[:, np.newaxis, :]),
            axis=2,
        ), )

    # Project each point back to the image using the camera.
    projections = project_point(camera, world_coords, num_frames)

    return (
        tf.transpose(projections, (1, 0, 2)),
        tf.transpose(depths),
        tf.transpose(world_coords, (1, 0, 2)),
    )


def estimate_occlusion_by_depth_and_segment(
        data,
        segments,
        x,
        y,
        num_frames,
        thresh,
        seg_id,
):
    """Estimate depth at a (floating point) x,y position.

    We prefer overestimating depth at the point, so we take the max over the 4
    neightoring pixels.

    Args:
      data: depth map. First axis is num_frames.
      segments: segmentation map. First axis is num_frames.
      x: x coordinate. First axis is num_frames.
      y: y coordinate. First axis is num_frames.
      num_frames: number of frames.
      thresh: Depth threshold at which we consider the point occluded.
      seg_id: Original segment id.  Assume occlusion if there's a mismatch.

    Returns:
      Depth for each point.
    """

    # need to convert from raster to pixel coordinates
    x = x - 0.5
    y = y - 0.5

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    shp = tf.shape(data)
    assert len(data.shape) == 3
    x0 = tf.clip_by_value(x0, 0, shp[2] - 1)
    x1 = tf.clip_by_value(x1, 0, shp[2] - 1)
    y0 = tf.clip_by_value(y0, 0, shp[1] - 1)
    y1 = tf.clip_by_value(y1, 0, shp[1] - 1)

    data = tf.reshape(data, [-1])
    rng = tf.range(num_frames)[:, tf.newaxis]
    i1 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
    i2 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
    i3 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
    i4 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

    depth = tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)

    segments = tf.reshape(segments, [-1])
    i1 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
    i2 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
    i3 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
    i4 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

    depth_occluded = tf.less(tf.transpose(depth), thresh)
    seg_occluded = True
    for i in [i1, i2, i3, i4]:
        i = tf.cast(i, tf.int32)
        seg_occluded = tf.logical_and(seg_occluded, tf.not_equal(seg_id, i))

    return tf.logical_or(depth_occluded, tf.transpose(seg_occluded))


def get_camera_matrices(
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        input_size,
        num_frames=None,
):
    """Tf function that converts camera positions into projection matrices."""
    intrinsics = []
    matrix_world = []
    assert cam_quaternions.shape[0] == num_frames
    for frame_idx in range(cam_quaternions.shape[0]):
        focal_length = tf.cast(cam_focal_length, tf.float32)
        sensor_width = tf.cast(cam_sensor_width, tf.float32)
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width * input_size[0] / input_size[1]
        p_x = 0.5
        p_y = 0.5
        intrinsics.append(
            tf.stack([
                tf.stack([f_x, 0., -p_x]),
                tf.stack([0., -f_y, -p_y]),
                tf.stack([0., 0., -1.]),
            ]))

        position = cam_positions[frame_idx]
        quat = cam_quaternions[frame_idx]
        rotation_matrix = from_quaternion(
            tf.concat([quat[1:], quat[0:1]], axis=0))
        transformation = tf.concat(
            [rotation_matrix, position[:, tf.newaxis]],
            axis=1,
        )
        transformation = tf.concat(
            [transformation,
             tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
            axis=0,
        )
        matrix_world.append(transformation)

    return (
        tf.cast(tf.stack(intrinsics), tf.float32),
        tf.cast(tf.stack(matrix_world), tf.float32),
    )


def quat2rot(quats):
    """Convert a list of quaternions to rotation matrices."""
    rotation_matrices = []
    for frame_idx in range(quats.shape[0]):
        quat = quats[frame_idx]
        rotation_matrix = from_quaternion(
            tf.concat([quat[1:], quat[0:1]], axis=0))
        rotation_matrices.append(rotation_matrix)
    return tf.cast(tf.stack(rotation_matrices), tf.float32)


def rotate_surface_normals(
        world_frame_normals,
        point_3d,
        cam_pos,
        obj_rot_mats,
        frame_for_query,
):
    """Points are occluded if the surface normal points away from the camera."""
    query_obj_rot_mat = tf.gather(obj_rot_mats, frame_for_query)
    obj_frame_normals = tf.einsum(
        'boi,bi->bo',
        tf.linalg.inv(query_obj_rot_mat),
        world_frame_normals,
    )
    world_frame_normals_frames = tf.einsum(
        'foi,bi->bfo',
        obj_rot_mats,
        obj_frame_normals,
    )
    cam_to_pt = point_3d - cam_pos[tf.newaxis, :, :]
    dots = tf.reduce_sum(world_frame_normals_frames * cam_to_pt, axis=-1)
    faces_away = dots > 0

    # If the query point also faces away, it's probably a bug in the meshes, so
    # ignore the result of the test.
    faces_away_query = tf.reduce_sum(
        tf.cast(faces_away, tf.int32)
        * tf.one_hot(frame_for_query, tf.shape(faces_away)[1], dtype=tf.int32),
        axis=1,
        keepdims=True,
    )
    faces_away = tf.logical_and(faces_away, tf.logical_not(faces_away_query > 0))
    return faces_away


# @tf.autograph.experimental.do_not_convert
def single_object_reproject(
        bbox_3d=None,
        pt=None,
        pt_segments=None,
        camera=None,
        cam_positions=None,
        num_frames=None,
        depth_map=None,
        segments=None,
        window=None,
        input_size=None,
        quat=None,
        normals=None,
        frame_for_pt=None,
        trust_normals=None,
):
    """Reproject points for a single object.

    Args:
      bbox_3d: The object bounding box from Kubric.  If none, assume it's
        background.
      pt: The set of points in 3D, with shape [num_points, 3]
      pt_segments: The segment each point came from, with shape [num_points]
      camera: Camera intrinsic parameters
      cam_positions: Camera positions, with shape [num_frames, 3]
      num_frames: Number of frames
      depth_map: Depth map video for the camera
      segments: Segmentation map video for the camera
      window: the window inside which we're sampling points
      input_size: [height, width] of the input images.
      quat: Object quaternion [num_frames, 4]
      normals: Point normals on the query frame [num_points, 3]
      frame_for_pt: Integer frame where the query point came from [num_points]
      trust_normals: Boolean flag for whether the surface normals for each query
        are trustworthy [num_points]

    Returns:
      Position for each point, of shape [num_points, num_frames, 2], in pixel
      coordinates, and an occlusion flag for each point, of shape
      [num_points, num_frames].  These are respect to the image frame, not the
      window.

    """
    # Finally, reproject
    reproj, depth_proj, world_pos = reproject(
        pt,
        camera,
        cam_positions,
        num_frames,
        bbox=bbox_3d,
    )

    occluded = tf.less(reproj[:, :, 2], 0)
    # TODO: input_size - 1
    reproj = reproj[:, :, 0:2] * np.array(input_size[::-1])[np.newaxis,
                                 np.newaxis, :]
    occluded = tf.logical_or(
        occluded,
        estimate_occlusion_by_depth_and_segment(
            depth_map[:, :, :, 0],
            segments[:, :, :, 0],
            tf.transpose(reproj[:, :, 0]),
            tf.transpose(reproj[:, :, 1]),
            num_frames,
            depth_proj * .99,
            pt_segments,
        ),
    )
    obj_occ = occluded
    obj_reproj = reproj

    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))

    if quat is not None:
        faces_away = rotate_surface_normals(
            normals,
            world_pos,
            cam_positions,
            quat2rot(quat),
            frame_for_pt,
        )
        faces_away = tf.logical_and(faces_away, trust_normals)
    else:
        # world is convex; can't face away from cam.
        faces_away = tf.zeros([tf.shape(pt)[0], num_frames], dtype=tf.bool)

    return obj_reproj, tf.logical_or(faces_away, obj_occ)


# @tf.autograph.experimental.do_not_convert
def track_points(
        object_coordinates,
        depth,
        depth_range,
        segmentations,
        surface_normals,
        bboxes_3d,
        obj_quat,
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        window,
        tracks_to_sample=256,
        sampling_stride=4,
        max_seg_id=25,
        max_sampled_frac=0.1,
):
    """Track points in 2D using Kubric data.

    Args:
      object_coordinates: Video of coordinates for each pixel in the object's
        local coordinate frame.  Shape [num_frames, height, width, 3]
      depth: uint16 depth video from Kubric.  Shape [num_frames, height, width]
      depth_range: Values needed to normalize Kubric's int16 depth values into
        metric depth.
      segmentations: Integer object id for each pixel.  Shape
        [num_frames, height, width]
      surface_normals: uint16 surface normal map. Shape
        [num_frames, height, width, 3]
      bboxes_3d: The set of all object bounding boxes from Kubric
      obj_quat: Quaternion rotation for each object.  Shape
        [num_objects, num_frames, 4]
      cam_focal_length: Camera focal length
      cam_positions: Camera positions, with shape [num_frames, 3]
      cam_quaternions: Camera orientations, with shape [num_frames, 4]
      cam_sensor_width: Camera sensor width parameter
      window: the window inside which we're sampling points.  Integer valued
        in the format [x_min, y_min, x_max, y_max], where min is inclusive and
        max is exclusive.
      tracks_to_sample: Total number of tracks to sample per video.
      sampling_stride: For efficiency, query points are sampled from a random grid
        of this stride.
      max_seg_id: The maxium segment id in the video.
      max_sampled_frac: The maximum fraction of points to sample from each
        object, out of all points that lie on the sampling grid.

    Returns:
      A set of queries, randomly sampled from the video (with a bias toward
        objects), of shape [num_points, 3].  Each point is [t, y, x], where
        t is time.  All points are in pixel/frame coordinates.
      The trajectory for each query point, of shape [num_points, num_frames, 3].
        Each point is [x, y].  Points are in pixel coordinates
      Occlusion flag for each point, of shape [num_points, num_frames].  This is
        a boolean, where True means the point is occluded.

    """
    extr_frame = 0
    # tf.print('SEGM MAX: ', tf.math.reduce_max(segmentations), output_stream=sys.stdout)

    chosen_points = []
    all_reproj = []
    all_occ = []

    # Convert to metric depth

    depth_range_f32 = tf.cast(depth_range, tf.float32)
    depth_min = depth_range_f32[0]
    depth_max = depth_range_f32[1]
    depth_f32 = tf.cast(depth, tf.float32)
    depth_map = depth_min + depth_f32 * (depth_max - depth_min) / 65535

    surface_normal_map = surface_normals / 65535 * 2. - 1.

    input_size = object_coordinates.shape.as_list()[1:3]
    num_frames = object_coordinates.shape.as_list()[0]

    # We first sample query points within the given window.  That means first
    # extracting the window from the segmentation tensor, because we want to have
    # a bias toward moving objects.
    # Note: for speed we sample points on a grid.  The grid start position is
    # randomized within the window. TODO: probably need for tf.zeros instead of tf.random.uniform([])
    start_vec = [
        # tf.random.uniform([], minval=0, maxval=sampling_stride, dtype=tf.int32)
        tf.zeros([], dtype=tf.int32)
        for _ in range(2)
    ]
    start_vec[0] += window[0]
    start_vec[1] += window[1]
    end_vec = [window[2], window[3]]

    # @tf.autograph.experimental.do_not_convert
    def extract_box(x, frame=0):
        x = x[frame, start_vec[0]:window[2]:sampling_stride,
            start_vec[1]:window[3]:sampling_stride]
        return x

    segmentations_box = extract_box(segmentations, frame=extr_frame)
    # tf.print('SEGM box shape: ', segmentations_box.shape, output_stream=sys.stdout)
    object_coordinates_box = extract_box(object_coordinates, frame=extr_frame)

    # num_to_sample.set_shape([max_seg_id])
    intrinsics, matrix_world = get_camera_matrices(
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        input_size,
        num_frames=num_frames,
    )

    # If the normal map is very rough, it's often because they come from a normal
    # map rather than the mesh.  These aren't trustworthy, and the normal test
    # may fail (i.e. the normal is pointing away from the camera even though the
    # point is still visible).  So don't use the normal test when inferring
    # occlusion.
    trust_sn = True
    sn_pad = tf.pad(surface_normal_map, [(0, 0), (1, 1), (1, 1), (0, 0)])
    shp = surface_normal_map.shape
    sum_thresh = 0
    for i in [0, 2]:
        for j in [0, 2]:
            diff = sn_pad[:, i: shp[1] + i, j: shp[2] + j, :] - surface_normal_map
            diff = tf.reduce_sum(tf.square(diff), axis=-1)
            sum_thresh += tf.cast(diff > 0.05 * 0.05, tf.int32)
    trust_sn = tf.logical_and(trust_sn, (sum_thresh <= 2))[..., tf.newaxis]
    surface_normals_box = extract_box(surface_normal_map)
    trust_sn_box = extract_box(trust_sn)

    def get_camera(fr=None):
        if fr is None:
            return {'intrinsics': intrinsics, 'matrix_world': matrix_world}
        return {'intrinsics': intrinsics[fr], 'matrix_world': matrix_world[fr]}

    # Construct pixel coordinates for each pixel within the window.
    window = tf.cast(window, tf.float32)
    y, x = tf.meshgrid(
        *[
            tf.range(st, ed, sampling_stride)
            for st, ed in zip(start_vec, end_vec)
        ],
        indexing='ij')
    z = tf.ones_like(x) * extr_frame
    pix_coords = tf.reshape(tf.stack([z, y, x], axis=-1), [-1, 3])

    for i in range(max_seg_id):
        # sample points on object i in the first frame.  obj_id is the position
        # within the object_coordinates array, which is one lower than the value
        # in the segmentation mask (0 in the segmentation mask is the background
        # object, which has no bounding box).
        obj_id = i - 1
        mask = tf.equal(tf.reshape(segmentations_box, [-1]), i)
        pt = tf.boolean_mask(tf.reshape(object_coordinates_box, [-1, 3]), mask)
        normals = tf.boolean_mask(tf.reshape(surface_normals_box, [-1, 3]), mask)
        trust_sn_mask = tf.boolean_mask(tf.reshape(trust_sn_box, [-1, 1]), mask)
        pt_coords = tf.boolean_mask(pix_coords, mask)
        trust_sn_gather = trust_sn_mask

        pixel_to_raster = tf.constant([0.0, 0.5, 0.5])[tf.newaxis, :]

        if obj_id == -1:
            # For the background object, no bounding box is available.  However,
            # this doesn't move, so we use the depth map to backproject these points
            # into 3D and use those positions throughout the video.
            pt_3d = []
            pt_coords_reorder = []
            for fr in range(num_frames):
                # We need to loop over frames because we need to use the correct depth
                # map for each frame.
                pt_coords_chunk = tf.boolean_mask(pt_coords,
                                                  tf.equal(pt_coords[:, 0], fr))
                pt_coords_reorder.append(pt_coords_chunk)

                pt_3d.append(
                    unproject(pt_coords_chunk[:, 1:], get_camera(fr), depth_map[fr]))
            pt = tf.concat(pt_3d, axis=0)
            chosen_points.append(
                tf.cast(tf.concat(pt_coords_reorder, axis=0), tf.float32) +
                pixel_to_raster)
            bbox = None
            quat = None
            frame_for_pt = None
        else:
            # For any other object, we just use the point coordinates supplied by
            # kubric.
            # pt = tf.gather(pt, idx)
            # # tf.print('PT1: ', pt, output_stream=sys.stdout)
            pt = pt / np.iinfo(np.uint16).max - .5
            chosen_points.append(tf.cast(pt_coords, tf.float32) + pixel_to_raster)
            # if obj_id>num_objects, then we won't have a box.  We also won't have
            # points, so just use a dummy to prevent tf from crashing.
            bbox = tf.cond(obj_id >= tf.shape(bboxes_3d)[0], lambda: bboxes_3d[0, :],
                           lambda: bboxes_3d[obj_id, :])
            quat = tf.cond(obj_id >= tf.shape(obj_quat)[0], lambda: obj_quat[0, :],
                           lambda: obj_quat[obj_id, :])
            frame_for_pt = pt_coords[..., 0]

        # Finally, compute the reprojections for this particular object.
        obj_reproj, obj_occ = tf.cond(
            tf.shape(pt)[0] > 0,
            functools.partial(
                single_object_reproject,
                bbox_3d=bbox,
                pt=pt,
                pt_segments=i,
                camera=get_camera(),
                cam_positions=cam_positions,
                num_frames=num_frames,
                depth_map=depth_map,
                segments=segmentations,
                window=window,
                input_size=input_size,
                quat=quat,
                normals=normals,
                frame_for_pt=frame_for_pt,
                trust_normals=trust_sn_gather,
            ),
            lambda:  # pylint: disable=g-long-lambda
            (tf.zeros([0, num_frames, 2], dtype=tf.float32),
             tf.zeros([0, num_frames], dtype=tf.bool)))
        all_reproj.append(obj_reproj)
        all_occ.append(obj_occ)

    # Points are currently in pixel coordinates of the original video.  We now
    # convert them to coordinates within the window frame, and rescale to
    # pixel coordinates.  Note that this produces the pixel coordinates after
    # the window gets cropped and rescaled to the full image size.
    wd = tf.concat(
        [np.array([0.0]), window[0:2],
         np.array([num_frames]), window[2:4]],
        axis=0)
    wd = wd[tf.newaxis, tf.newaxis, :]
    coord_multiplier = [num_frames, input_size[0], input_size[1]]
    all_reproj = tf.concat(all_reproj, axis=0)
    # We need to extract x,y, but the format of the window is [t1,y1,x1,t2,y2,x2]
    window_size = wd[:, :, 5:3:-1] - wd[:, :, 2:0:-1]
    window_top_left = wd[:, :, 2:0:-1]
    all_reproj = (all_reproj - window_top_left) / window_size
    all_reproj = all_reproj * coord_multiplier[2:0:-1]
    all_occ = tf.concat(all_occ, axis=0)

    # chosen_points is [num_points, (z,y,x)]
    chosen_points = tf.concat(chosen_points, axis=0)

    chosen_points = tf.cast(chosen_points, tf.float32)

    # renormalize so the box corners are at [-1,1]
    chosen_points = (chosen_points - wd[:, 0, :3]) / (wd[:, 0, 3:] - wd[:, 0, :3])
    chosen_points = chosen_points * coord_multiplier
    # Note: all_reproj is in (x,y) format, but chosen_points is in (z,y,x) format

    return tf.cast(chosen_points, tf.float32), tf.cast(all_reproj,
                                                       tf.float32), all_occ


def _get_distorted_bounding_box(
        jpeg_shape,
        bbox,
        min_object_covered,
        aspect_ratio_range,
        area_range,
        max_attempts,
):
    """Sample a crop window to be used for cropping."""
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        jpeg_shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack(
        [offset_y, offset_x, offset_y + target_height, offset_x + target_width])
    return crop_window


@tf.autograph.experimental.do_not_convert
def add_tracks(data,
               train_size=(256, 256),
               vflip=False,
               random_crop=True,
               tracks_to_sample=256,
               sampling_stride=4,
               max_seg_id=25,
               max_sampled_frac=0.1):
    """Track points in 2D using Kubric data.

    Args:
      data: Kubric data, including RGB/depth/object coordinate/segmentation
        videos and camera parameters.
      train_size: Cropped output will be at this resolution.  Ignored if
        random_crop is False.
      vflip: whether to vertically flip images and tracks (to test generalization)
      random_crop: Whether to randomly crop videos
      tracks_to_sample: Total number of tracks to sample per video.
      sampling_stride: For efficiency, query points are sampled from a random grid
        of this stride.
      max_seg_id: The maxium segment id in the video.
      max_sampled_frac: The maximum fraction of points to sample from each
        object, out of all points that lie on the sampling grid.

    Returns:
      A dict with the following keys:
      query_points:
        A set of queries, randomly sampled from the video (with a bias toward
        objects), of shape [num_points, 3].  Each point is [t, y, x], where
        t is time.  Points are in pixel/frame coordinates.
        [num_frames, height, width].
      target_points:
        The trajectory for each query point, of shape [num_points, num_frames, 3].
        Each point is [x, y].  Points are in pixel/frame coordinates.
      occlusion:
        Occlusion flag for each point, of shape [num_points, num_frames].  This is
        a boolean, where True means the point is occluded.
      video:
        The cropped video, normalized into the range [-1, 1]

    """
    shp = data['video'].shape.as_list()
    num_frames = shp[0]
    # if any([s % sampling_stride != 0 for s in shp[:-1]]):
    #     raise ValueError('All video dims must be a multiple of sampling_stride.')

    crop_window = tf.constant([0, 0, shp[1], shp[2]],
                              dtype=tf.int32,
                              shape=[4])

    query_points, target_points, occluded = track_points(
        data['object_coordinates'], data['depth'],
        data['metadata']['depth_range'], data['segmentations'],
        data['normal'],
        data['instances']['bboxes_3d'], data['instances']['quaternions'],
        data['camera']['focal_length'],
        data['camera']['positions'], data['camera']['quaternions'],
        data['camera']['sensor_width'], crop_window, tracks_to_sample,
        sampling_stride, max_seg_id, max_sampled_frac)
    video = data['video']

    # print('POINTS: ', query_points.shape, target_points.shape, occluded.shape)

    shp = video.shape.as_list()
    query_points.set_shape([None, 3])
    target_points.set_shape([None, num_frames, 2])
    occluded.set_shape([None, num_frames])

    # Crop the video to the sampled window, in a way which matches the coordinate
    # frame produced the track_points functions.
    crop_window = crop_window / (
            np.array(shp[1:3] + shp[1:3]).astype(np.float32) - 1)
    crop_window = tf.tile(crop_window[tf.newaxis, :], [num_frames, 1])
    video = tf.image.crop_and_resize(
        video,
        tf.cast(crop_window, tf.float32),
        tf.range(num_frames),
        train_size,
    )
    if vflip:
        video = video[:, ::-1, :, :]
        target_points = target_points * np.array([1, -1])
        query_points = query_points * np.array([1, -1, 1])
    res = {
        'query_points': query_points,
        'target_points': target_points,
        'occluded': occluded,
        'video': video / (255. / 2.) - 1.,
    }
    return res


def create_point_tracking_dataset(
        train_size=(256, 256),
        shuffle_buffer_size=256,
        split='train',
        batch_dims=tuple(),
        repeat=True,
        vflip=False,
        random_crop=False,
        tracks_to_sample=4,
        sampling_stride=1,
        max_seg_id=25,
        max_sampled_frac=1.0,
        num_parallel_point_extraction_calls=16,
        **kwargs):
    """Construct a dataset for point tracking using Kubric: go/kubric.

    Args:
      train_size: Tuple of 2 ints. Cropped output will be at this resolution
      shuffle_buffer_size: Int. Size of the shuffle buffer
      split: Which split to construct from Kubric.  Can be 'train' or
        'validation'.
      batch_dims: Sequence of ints. Add multiple examples into a batch of this
        shape.
      repeat: Bool. whether to repeat the dataset.
      vflip: Bool. whether to vertically flip the dataset to test generalization.
      random_crop: Bool. whether to randomly crop videos
      tracks_to_sample: Int. Total number of tracks to sample per video.
      sampling_stride: Int. For efficiency, query points are sampled from a
        random grid of this stride.
      max_seg_id: Int. The maxium segment id in the video.  Note the size of
        the to graph is proportional to this number, so prefer small values.
      max_sampled_frac: Float. The maximum fraction of points to sample from each
        object, out of all points that lie on the sampling grid.
      num_parallel_point_extraction_calls: Int. The num_parallel_calls for the
        map function for point extraction.
      **kwargs: additional args to pass to tfds.load.

    Returns:
      The dataset generator.
    """

    data_name = kwargs.get('name', 'movi_e/256x256')
    data_dir = kwargs.get('data_dir', 'gs://kubric-public/tfds')

    ds = tfds.load(
        data_name,
        data_dir=data_dir,
        shuffle_files=shuffle_buffer_size is not None,
        **kwargs)

    ds = ds[split]
    if repeat:
        ds = ds.repeat()
    ds = ds.map(
        functools.partial(
            add_tracks,
            train_size=train_size,
            vflip=vflip,
            random_crop=random_crop,
            tracks_to_sample=tracks_to_sample,
            sampling_stride=sampling_stride,
            max_seg_id=max_seg_id,
            max_sampled_frac=max_sampled_frac),
        num_parallel_calls=num_parallel_point_extraction_calls)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    for bs in batch_dims[::-1]:
        ds = ds.batch(bs)

    return ds


def plot_tracks(rgb, points, occluded, trackgroup=None):
    """Plot tracks with matplotlib."""
    disp = []
    cmap = plt.cm.hsv

    z_list = np.arange(
        points.shape[0]) if trackgroup is None else np.array(trackgroup)
    # random permutation of the colors so nearby points in the list can get
    # different colors
    z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
    colors = cmap(z_list / (np.max(z_list) + 1))
    figure_dpi = 64

    for i in range(rgb.shape[0]):
        fig = plt.figure(
            figsize=(256 / figure_dpi, 256 / figure_dpi),
            dpi=figure_dpi,
            frameon=False,
            facecolor='w')
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(rgb[i])

        valid = points[:, i, 0] > 0
        valid = np.logical_and(valid, points[:, i, 0] < rgb.shape[2] - 1)
        valid = np.logical_and(valid, points[:, i, 1] > 0)
        valid = np.logical_and(valid, points[:, i, 1] < rgb.shape[1] - 1)

        colalpha = np.concatenate([colors[:, :-1], 1 - occluded[:, i:i + 1]],
                                  axis=1)
        # Note: matplotlib uses pixel corrdinates, not raster.
        plt.scatter(
            points[valid, i, 0] - 0.5,
            points[valid, i, 1] - 0.5,
            s=3,
            c=colalpha[valid],
        )

        occ2 = occluded[:, i:i + 1]

        colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

        plt.scatter(
            points[valid, i, 0],
            points[valid, i, 1],
            s=20,
            facecolors='none',
            edgecolors=colalpha[valid],
        )

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(int(height), int(width), 3)
        disp.append(np.copy(img))
        plt.close(fig)

    return np.stack(disp, axis=0)


def save_flowou(rgb, target_points, query_points, occluded, file_num, split,
                save_root='datasets/kubric_movi_e_longterm'):
    N, H, W, C = rgb.shape

    occls = np.zeros([H, W, N], dtype=occluded.dtype)
    positions = np.zeros([H, W, N, 2], dtype=np.float32)

    c_save_root = os.path.join(save_root, split, f'{file_num:05d}')

    # TODO: jak se zbavit for-loop?
    for idx in range(H * W):
        idx_h, idx_w = np.round(query_points[idx, 1:] - 0.5).astype(int)
        positions[idx_h, idx_w, :, :] = target_points[idx, :, :] - 0.5
        occls[idx_h, idx_w, :] = occluded[idx, :]

    for frame_idx in range(N):
        # NOTE: clipped because of assertion in saving flowou
        # sometimes the flow is greater than resolution of images
        c_rgb = np.clip(255 * ((rgb[frame_idx, :, :, :] * 0.5) + 0.5), 0, 255).astype(np.uint8)
        c_rgb_path = os.path.join(c_save_root, 'images', f'{frame_idx:04d}.png')
        mkdir_from_full_file_path_if_not_exist(c_rgb_path)
        cv2.imwrite(c_rgb_path, cv2.cvtColor(c_rgb, cv2.COLOR_RGB2BGR))

    x0, y0 = np.meshgrid(np.arange(W), np.arange(H))
    position_zero = np.stack([x0, y0], axis=2).astype(float)
    for frame_idx in range(N):
        c_pos = positions[:, :, frame_idx, :]
        c_flow = c_pos - position_zero
        c_flow = np.clip(c_flow, -(2 ** 15 / 32 - 1), (2 ** 15 / 32 - 1))
        c_flow_reshape = einops.rearrange(c_flow, 'H W xy -> xy H W', xy=2)

        # print('FLOW MIN/MAX: ', c_flow.min(), c_flow.max())

        c_occl = occls[:, :, frame_idx]
        c_occl_reshape = einops.rearrange(c_occl, 'H W -> 1 H W')

        c_unc_dummy = np.zeros_like(c_occl_reshape)

        c_flowou_path = os.path.join(c_save_root, 'flowou', f'{0:04d}_to_{frame_idx:04d}.flowou.png')
        mkdir_from_full_file_path_if_not_exist(c_flowou_path)
        write_flowou_png(c_flowou_path, c_flow_reshape, c_occl_reshape, c_unc_dummy)


def main(split='validation'):
    ds = tfds.as_numpy(create_point_tracking_dataset(shuffle_buffer_size=None, repeat=False, split=split))

    # print('dataset size: ', len(ds))
    for i, data in enumerate(ds):
        save_flowou(data['video'], data['target_points'], data['query_points'], data['occluded'],
                    file_num=i, split=split)


def get_flow(data_name=None, data_dir=None, query_frame=None, target_frame=None):
    ds = tfds.as_numpy(create_point_tracking_dataset(
        data_dir='datasets/kubric_longtermflow_dataset/',
        name='00001/RES_1024x1024__FPS_12__NFRAMES_24',
        shuffle_buffer_size=None,
        repeat=False,
        split='validation'))
    for i, data in enumerate(ds):
        save_flowou(data['video'], data['target_points'], data['query_points'], data['occluded'],
                    file_num=i, split='validation')


from etils import epath

# PathLike = Union[str, tfds.core.ReadWritePath]
PathLike = Union[str, epath.Path]


def as_path(path: PathLike):  # -> tfds.core.ReadWritePath:
    """Convert str or pathlike object to tfds.core.ReadWritePath.

    Instead of pathlib.Paths, we use the TFDS path because they transparently
    support paths to GCS buckets such as "gs://kubric-public/GSO".
    """
    return epath.Path(path)


def read_png(filename, rescale_range=None) -> np.ndarray:
    filename = as_path(filename)
    png_reader = png.Reader(bytes=filename.read_bytes())
    width, height, pngdata, info = png_reader.read()
    del png_reader

    bitdepth = info["bitdepth"]
    if bitdepth == 8:
        dtype = np.uint8
    elif bitdepth == 16:
        dtype = np.uint16
    else:
        raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")

    plane_count = info["planes"]
    pngdata = np.vstack(list(map(dtype, pngdata)))
    if rescale_range is not None:
        minv, maxv = rescale_range
        pngdata = pngdata / 2 ** bitdepth * (maxv - minv) + minv

    return pngdata.reshape((height, width, plane_count))


def read_tiff(filename: PathLike) -> np.ndarray:
    filename = as_path(filename)
    img = imageio.v2.imread(filename.read_bytes(), format="tiff")
    if img.ndim == 2:
        img = img[:, :, None]
    return img


def format_instance_information(obj):
    return {
        "mass": obj["mass"],
        "friction": obj["friction"],
        "restitution": obj["restitution"],
        "positions": np.array(obj["positions"], np.float32),
        "quaternions": np.array(obj["quaternions"], np.float32),
        "velocities": np.array(obj["velocities"], np.float32),
        "angular_velocities": np.array(obj["angular_velocities"], np.float32),
        "bboxes_3d": np.array(obj["bboxes_3d"], np.float32),
        "image_positions": np.array(obj["image_positions"], np.float32),
        "bboxes": [tfds.features.BBox(*bbox) for bbox in obj["bboxes"]],
        "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
        "visibility": np.array(obj["visibility"], dtype=np.uint16),
    }


def format_camera_information(metadata):
    return {
        "focal_length": metadata["camera"]["focal_length"],
        "sensor_width": metadata["camera"]["sensor_width"],
        "field_of_view": metadata["camera"]["field_of_view"],
        "positions": np.array(metadata["camera"]["positions"], np.float32),
        "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
    }


def get_events_features():
    return {
        "collisions": tfds.features.Sequence({
            "instances": tfds.features.Tensor(shape=(2,), dtype=tf.uint16),
            "frame": tf.int32,
            "force": tf.float32,
            "position": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            "image_position": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            "contact_normal": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        })
    }


def format_events_information(events):
    return {
        "collisions": [{
            "instances": np.array(c["instances"], dtype=np.uint16),
            "frame": c["frame"],
            "force": c["force"],
            "position": np.array(c["position"], dtype=np.float32),
            "image_position": np.array(c["image_position"], dtype=np.float32),
            "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
        } for c in events["collisions"]],
    }


def convert_float_to_uint16(array, min_val, max_val):
    return np.round((array - min_val) / (max_val - min_val) * 65535
                    ).astype(np.uint16)


def filter_frames(data_list, frames=None, index=None):
    if frames is None:
        return data_list
    if index is None:
        return [data_list[cf] for cf in frames]
    elif index == 0:
        data_list = [data_list[cf] for cf in frames]
    elif index == 1:
        data_list = [data_list[:, cf] for cf in frames]
    elif index == 2:
        data_list = [data_list[:, :, cf] for cf in frames]
    else:
        raise NotImplementedError(f'for index {index}')
    return np.stack(data_list, index)


def convert2tensorflow(listd, frames=None):
    dataout = {}
    if isinstance(listd, list):
        keys = listd[0].keys()
        # listd = filter_frames(listd, frames)
        for k in keys:
            if k in ['bboxes', 'bbox_frames']:
                continue
            c_list = [c_l[k] for c_l in listd]
            c_data = np.stack(c_list, 0)
            if k in ['positions', 'quaternions', 'velocities', 'angular_velocities', 'bboxes_3d', 'image_positions',
                     'visibility']:
                c_data = filter_frames(c_data, frames=frames, index=1)
            dataout[k] = tf.convert_to_tensor(c_data)
    else:
        keys = listd.keys()
        for k in keys:
            if k in ['collisions']:
                continue
            if (k in ['positions', 'quaternions']) and (frames is not None):
                c_data = filter_frames(listd[k], frames)
                dataout[k] = tf.convert_to_tensor(np.stack(c_data, 0))
            else:
                dataout[k] = tf.convert_to_tensor(listd[k])

    return dataout


def load_scene_directory(scene_dir, layers=DEFAULT_LAYERS, frames=None):
    scene_dir = as_path(scene_dir)
    example_key = f"{scene_dir.name}"

    with tf.io.gfile.GFile(str(scene_dir / "data_ranges.json"), "r") as fp:
        data_ranges = json.load(fp)

    with tf.io.gfile.GFile(str(scene_dir / "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    with tf.io.gfile.GFile(str(scene_dir / "events.json"), "r") as fp:
        events = json.load(fp)

    num_frames = metadata["metadata"]["num_frames"]
    resolution = metadata["metadata"]["resolution"]
    result = {
        "metadata": {
            "video_name": example_key,
            "width": resolution[1],  # FIXME: here may be a bug - switched coordinates (hard to say when W=H)
            "height": resolution[0],
            "num_frames": num_frames if frames is not None else len(frames),
            "num_instances": metadata["metadata"]["num_instances"],
            # "motion_blur": metadata["metadata"]["motion_blur"]
        },
        "background": metadata["metadata"]["background"],
        "instances": convert2tensorflow([format_instance_information(obj) for obj in metadata["instances"]],
                                        frames=frames),
        "camera": convert2tensorflow(format_camera_information(metadata), frames=frames),
        "events": convert2tensorflow(format_events_information(events), frames=frames),
    }

    metadata['metadata']['height'] = metadata['metadata']['resolution'][0]
    metadata['metadata']['width'] = metadata['metadata']['resolution'][1]

    paths = {
        key: filter_frames([scene_dir / f"{key}_{f:05d}.png" for f in range(num_frames)], frames)
        for key in layers if key != "depth"
    }

    if "depth" in layers:
        depth_paths = [scene_dir / f"depth_{f:05d}.tiff" for f in range(num_frames)]
        depth_paths = filter_frames(depth_paths, frames)
        depth_frames = np.array([read_tiff(frame_path) for frame_path in depth_paths])
        depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
        result["depth"] = tf.convert_to_tensor(np.stack(convert_float_to_uint16(depth_frames, depth_min, depth_max), 0))
        result["metadata"]["depth_range"] = [depth_min, depth_max]

    # if "forward_flow" in layers:
    #   result["metadata"]["forward_flow_range"] = [
    #       data_ranges["forward_flow"]["min"] / scale * 512,
    #       data_ranges["forward_flow"]["max"] / scale * 512]
    #   result["forward_flow"] = [read_png(frame_path)[..., :2] for frame_path in paths["forward_flow"]]
    #
    # if "backward_flow" in layers:
    #   result["metadata"]["backward_flow_range"] = [
    #       data_ranges["backward_flow"]["min"] / scale * 512,
    #       data_ranges["backward_flow"]["max"] / scale * 512]
    #   result["backward_flow"] = [read_png(frame_path)[..., :2] for frame_path in paths["backward_flow"]]

    # for key in ["normal", "object_coordinates", "uv"]:
    for key in ["normal", "object_coordinates"]:
        if key in layers:
            result[key] = tf.convert_to_tensor(np.stack([read_png(frame_path) for frame_path in paths[key]], 0))

    if "segmentation" in layers:
        # somehow we ended up calling this "segmentations" in TFDS and
        # "segmentation" in kubric. So we have to treat it separately.
        result["segmentations"] = tf.convert_to_tensor(
            np.stack([read_png(frame_path) for frame_path in paths["segmentation"]], 0))

    if "rgba" in layers:
        result["video"] = tf.convert_to_tensor(
            np.stack([read_png(frame_path)[..., :3] for frame_path in paths["rgba"]], 0))

    return example_key, result, metadata


def get_flow_from_points(rgb, target_points, query_points, occluded):
    N, H, W, C = rgb.shape

    occls = np.zeros([H, W, N], dtype=bool)
    positions = np.zeros([H, W, N, 2], dtype=np.float32)

    occl_list = []
    flow_list = []
    rgb_list = []

    query_points = np.round(query_points - 0.5).astype(int)
    positions[query_points[:, 1], query_points[:, 2], :, :] = target_points[:, :, :] - 0.5
    occls[query_points[:, 1], query_points[:, 2], :] = occluded[:, :]

    for frame_idx in range(N):
        # NOTE: clipped because of assertion in saving flowou
        # sometimes the flow is greater than resolution of images
        c_rgb = np.clip(255 * ((rgb[frame_idx, :, :, :] * 0.5) + 0.5), 0, 255).astype(np.uint8)
        rgb_list.append(c_rgb)

    x0, y0 = np.meshgrid(np.arange(W), np.arange(H))
    position_zero = np.stack([x0, y0], axis=2).astype(float)
    for frame_idx in range(N):
        c_pos = positions[:, :, frame_idx, :]
        c_flow = c_pos - position_zero
        c_flow_reshape = einops.rearrange(c_flow, 'H W xy -> xy H W', xy=2)
        flow_list.append(c_flow_reshape)

        c_occl = occls[:, :, frame_idx]
        c_occl_reshape = einops.rearrange(c_occl, 'H W -> 1 H W')
        occl_list.append(c_occl_reshape)
    return {'rgb': rgb_list, 'occlusion': occl_list, 'flow': flow_list}


def get_multiflow(dir_path, frames=None):
    """Read data from dir path and create optical flow and occlusion for specified frames or for all frames

    Args:
        dir_path: location of custom kubric data directory
                    e.g. 'datasets/kubric_longtermflow_dataset/00715/RES_1024x1024__FPS_120__NFRAMES_240'
        frames: list of frames for which the optical flow and occlusion will be created,
                    source frame is always first in the list, target frames -- rest
                    default - None - from frame 0 to all other frames
                    e.g. None, [10, 100]

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """

    # load data from directory and convert them for tensorflow
    ds = load_scene_directory(dir_path, frames=frames)[1]

    train_size = (ds['metadata']['height'], ds['metadata']['width'])
    vflip = False
    random_crop = False
    tracks_to_sample = 4
    sampling_stride = 1
    max_seg_id = 25
    max_sampled_frac = 1.0

    # create tracking dataset from first (or first in a frames list) to every other frame
    data = add_tracks(ds,
                      train_size=train_size,
                      vflip=vflip,
                      random_crop=random_crop,
                      tracks_to_sample=tracks_to_sample,
                      sampling_stride=sampling_stride,
                      max_seg_id=max_seg_id,
                      max_sampled_frac=max_sampled_frac)

    # convert tracking to flow and occlusion
    return get_flow_from_points(data['video'], data['target_points'], data['query_points'], data['occluded'])


if __name__ == '__main__':
    main(split='train')
