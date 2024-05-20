# python3.5, TensorFlow 1.9
from __future__ import print_function, division
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import fnmatch
import copy

import tensorflow as tf
import png
import numpy as np

import fnmatch
import os
import copy

import re
import numpy as np
from struct import *

from os import listdir
from os.path import isfile


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', relative=True, **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            scale = 1.0
            if relative:
                scale = _width_f

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * scale

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _num_batch = input_images.get_shape().as_list()[0]
        _height = input_images.get_shape().as_list()[1]
        _width = input_images.get_shape().as_list()[2]
        _num_channels = input_images.get_shape().as_list()[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output


def bilinear_sampler_2d_h(input_images, offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)
            y = tf.clip_by_value(y, 0.0, _height_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * _edge_size), tf.int32)
            y1 = tf.cast(tf.minimum(y1_f, _height_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_00 = base_y0 + x0
            idx_01 = base_y1 + x0
            idx_10 = base_y0 + x1
            idx_11 = base_y1 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_00 = tf.gather(im_flat, idx_00)
            pix_01 = tf.gather(im_flat, idx_01)
            pix_10 = tf.gather(im_flat, idx_10)
            pix_11 = tf.gather(im_flat, idx_11)

            weight_00 = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
            weight_01 = tf.expand_dims((x1_f - x) * (y - y0_f), 1)
            weight_10 = tf.expand_dims((x - x0_f) * (y1_f - y), 1)
            weight_11 = tf.expand_dims((x - x0_f) * (y - y0_f), 1)

            return weight_00 * pix_00 \
                + weight_01 * pix_01 \
                + weight_10 * pix_10 \
                + weight_11 * pix_11

    def _transform(input_images, x_offset, y_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f
            y_t_flat = y_t_flat + tf.reshape(y_offset, [-1]) * _height_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    def _expand(x):
        x_batch = tf.expand_dims(x, 0)
        x_batch = tf.transpose(x_batch, [0, 1, 3, 2])
        return x_batch

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _num_batch = input_images.get_shape().as_list()[0]
        _height = input_images.get_shape().as_list()[1]
        _width = input_images.get_shape().as_list()[2]
        _num_channels = input_images.get_shape().as_list()[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        # file_cond = tf.equal(tf.size(tf.shape(offset)), tf.constant(4, tf.int32))
        # offset = tf.cond(file_cond, lambda: offset, lambda: _expand(offset))
        offset = _expand(offset)

        x_offset = tf.slice(offset, [0, 0, 0, 0], [-1, -1, -1, 1])
        y_offset = tf.slice(offset, [0, 0, 0, 1], [-1, -1, -1, 1])

        output = _transform(input_images, x_offset, y_offset)

        return output


def bilinear_sampler_2d_absolute(input_data, offset, wrap_mode=None, name='bilinear_sampler'):
    if wrap_mode is not None:
        NotImplementedError('wrap mode is not implemented, use bilinear_sampler_2d_absolute_old() instead')
    if (input_data.shape[2].value is None) or (input_data.shape[1].value is None):
        return bilinear_sampler_2d_absolute_old(input_data, offset)
    with tf.variable_scope(name):
        unpacked_flow = tf.unstack(offset, axis=3)
        flow = tf.stack([-unpacked_flow[1], -unpacked_flow[0]], axis=3)
        return tf.contrib.image.dense_image_warp(input_data, flow)


def bilinear_sampler_2d_absolute_old(input_images, offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)
            y = tf.clip_by_value(y, 0.0, _height_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * _edge_size), tf.int32)
            y1 = tf.cast(tf.minimum(y1_f, _height_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_00 = base_y0 + x0
            idx_01 = base_y1 + x0
            idx_10 = base_y0 + x1
            idx_11 = base_y1 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_00 = tf.gather(im_flat, idx_00)
            pix_01 = tf.gather(im_flat, idx_01)
            pix_10 = tf.gather(im_flat, idx_10)
            pix_11 = tf.gather(im_flat, idx_11)

            weight_00 = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
            weight_01 = tf.expand_dims((x1_f - x) * (y - y0_f), 1)
            weight_10 = tf.expand_dims((x - x0_f) * (y1_f - y), 1)
            weight_11 = tf.expand_dims((x - x0_f) * (y - y0_f), 1)

            return weight_00 * pix_00 \
                + weight_01 * pix_01 \
                + weight_10 * pix_10 \
                + weight_11 * pix_11

    def _transform(input_images, x_offset, y_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1])
            y_t_flat = y_t_flat + tf.reshape(y_offset, [-1])

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        # _num_channels = tf.shape(input_images)[3]

        # _num_batch    = input_images.get_shape().as_list()[0]
        # _height       = input_images.get_shape().as_list()[1]
        # _width        = input_images.get_shape().as_list()[2]
        _num_channels = input_images.get_shape().as_list()[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        x_offset = tf.slice(offset, [0, 0, 0, 0], [-1, -1, -1, 1])
        y_offset = tf.slice(offset, [0, 0, 0, 1], [-1, -1, -1, 1])

        output = _transform(input_images, x_offset, y_offset)

        return output


def nearest_neighbour_sampler_2d_absolute(input_images, offset, round_mode='ceilceil', wrap_mode='border',
                                          name='nearest_neighbour_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)
            y = tf.clip_by_value(y, 0.0, _height_f - 1 + 2 * _edge_size)

            x0_f = tf.round(x)
            y0_f = tf.round(y)

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_00 = base_y0 + x0

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_00 = tf.gather(im_flat, idx_00)

            return pix_00

    def _transform(input_images, x_offset, y_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1])
            y_t_flat = y_t_flat + tf.reshape(y_offset, [-1])

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        # _num_batch    = input_images.get_shape().as_list()[0]
        # _height       = input_images.get_shape().as_list()[1]
        # _width        = input_images.get_shape().as_list()[2]
        # _num_channels = input_images.get_shape().as_list()[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        x_offset = tf.slice(offset, [0, 0, 0, 0], [-1, -1, -1, 1])
        y_offset = tf.slice(offset, [0, 0, 0, 1], [-1, -1, -1, 1])

        if round_mode == 'ceilceil':
            x_offset = tf.ceil(x_offset)
            y_offset = tf.ceil(y_offset)
        elif round_mode == 'ceilfloor':
            x_offset = tf.ceil(x_offset)
            y_offset = tf.floor(y_offset)
        elif round_mode == 'floorceil':
            x_offset = tf.floor(x_offset)
            y_offset = tf.ceil(y_offset)
        elif round_mode == 'floorfloor':
            x_offset = tf.floor(x_offset)
            y_offset = tf.floor(y_offset)
        else:
            NotImplementedError('this mode is not implemented')

        output = _transform(input_images, x_offset, y_offset)

        return output


# Original code snippet by https://lmb.informatik.uni-freiburg.de :
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def readPFMpython3(filename):
    file = open(filename, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('latin-1').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('latin-1').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



def read_pgm(filename):
    """Return a raster of integers from a PGM as a list of lists."""
    with open(filename, 'rb') as pgmf:
        assert pgmf.readline() == 'P5\n'
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())
        assert depth <= 255

        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)
        return np.asarray(raster)


def create_dir_if_no_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    filelists = []
    filelists_right = []
    flowlists = []
    displists = []

    filelists_back = []
    flowlists_back = []

    i = 0

    for root, dirnames, filenames in os.walk('frames_cleanpass/'):
        filelist = []
        displist = []
        filelist_right = []
        print(root)
        # print(filenames)

        for filename in fnmatch.filter(filenames, '*.png'):
            root_split = os.path.join(root, filename).split('/')
            root_split = root_split[1:]

            if root_split[-2] == 'left':
                filelist.append(os.path.join('frames_cleanpass', *root_split))
                root_split_copy = root_split[:]
                root_split[-1] = root_split[-1].replace('.png', ".pfm")

                root_split_copy[-2] = root_split_copy[-2].replace('left', "right")
                filelist_right.append(os.path.join('frames_cleanpass', *root_split_copy))

        if len(filelist) > 0:
            filelist.sort()
            filelists.append(filelist)

            filelist_right.sort()
            filelists_right.append(filelist_right)

            # Flow future
            root_split = root.split('/')
            root_split = root_split[1:]
            root_split.insert(-1, 'into_future')
            flowroot = os.path.join('optical_flow', *root_split)

            flowpath = flowroot + '/'
            flowpath_no_root = os.path.join('optical_flow', *root_split) + '/'
            onlyfiles = [os.path.join(flowpath_no_root, f) for f in listdir(flowpath) if isfile(os.path.join(flowpath, f))]
            onlyfiles.sort()
            flowlists.append(onlyfiles)

            # Flow back
            root_split_back = root.split('/')
            root_split_back = root_split_back[1:]
            root_split_back.insert(-1, 'into_past')
            flowroot_back = os.path.join('optical_flow', *root_split_back)

            flowpath_back = flowroot_back + '/'
            flowpath_no_root_back = os.path.join('optical_flow', *root_split_back) + '/'
            onlyfiles_back = [os.path.join(flowpath_no_root_back, f) for f in listdir(flowpath_back) if
                              isfile(os.path.join(flowpath_back, f))]
            onlyfiles_back.sort()
            flowlists_back.append(onlyfiles_back)


    boundaries_lists = []
    boundaries_lists_back = []
    for flow_list in flowlists:
        boundaries_list = [
            f.replace('optical_flow/', 'motion_boundaries/').replace('_R.pfm', '.pfm').replace('_L.pfm', '.pgm').replace(
                'OpticalFlowIntoFuture_', '').replace('OpticalFlowIntoPast_', '') for f in flow_list]
        boundaries_lists.append(boundaries_list)

    for flow_list in flowlists_back:
        boundaries_list = [
            f.replace('optical_flow/', 'motion_boundaries/').replace('_R.pfm', '.pfm').replace('_L.pfm', '.pgm').replace(
                'OpticalFlowIntoFuture_', '').replace('OpticalFlowIntoPast_', '') for f in flow_list]
        boundaries_lists_back.append(boundaries_list)


    n_files = 2
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        flow_t0 = tf.placeholder(dtype=tf.float32, shape=(540, 960, 3))
        flow_t0.set_shape([540, 960, 3])
        flow_t1 = tf.placeholder(dtype=tf.float32, shape=(540, 960, 3))
        flow_t1.set_shape([540, 960, 3])

        object_index_t0 = tf.placeholder(dtype=tf.float32, shape=(540, 960))
        object_index_t0.set_shape([540, 960])
        object_index_t1 = tf.placeholder(dtype=tf.float32, shape=(540, 960))
        object_index_t1.set_shape([540, 960])

        flow_t0_ex = tf.expand_dims(flow_t0, 0)
        flow_t1_ex = tf.expand_dims(flow_t1, 0)

        flow_t0_ex = tf.clip_by_value(flow_t0_ex, -1000.0, 1000.0)
        flow_t1_ex = tf.clip_by_value(flow_t1_ex, -1000.0, 1000.0)

        object_index_t0_ex = tf.expand_dims(object_index_t0, 0)
        object_index_t0_ex = tf.expand_dims(object_index_t0_ex, 3)
        object_index_t1_ex = tf.expand_dims(object_index_t1, 0)
        object_index_t1_ex = tf.expand_dims(object_index_t1_ex, 3)

        flow_t1_2_t0 = bilinear_sampler_2d_absolute(flow_t1_ex, flow_t0_ex)
        flow_t0_2_t1 = bilinear_sampler_2d_absolute(flow_t0_ex, flow_t1_ex)
        object_index_t1_2_t0 = bilinear_sampler_2d_absolute(object_index_t1_ex, flow_t0_ex)
        object_index_t0_2_t1 = bilinear_sampler_2d_absolute(object_index_t0_ex, flow_t1_ex)

        # COMPUTE IF OBJECT INDEX IS NOT ON THE BOUNDARY
        object_index_t1_2_t0_0 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t1_ex, flow_t0_ex, relative=False,
                                                     round_mode='ceilceil') - object_index_t0_ex)
        object_index_t1_2_t0_1 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t1_ex, flow_t0_ex, relative=False,
                                                     round_mode='ceilfloor') - object_index_t0_ex)
        object_index_t1_2_t0_2 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t1_ex, flow_t0_ex, relative=False,
                                                     round_mode='floorceil') - object_index_t0_ex)
        object_index_t1_2_t0_3 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t1_ex, flow_t0_ex, relative=False,
                                                     round_mode='floorfloor') - object_index_t0_ex)

        object_index_t0_2_t1_0 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t0_ex, flow_t1_ex, relative=False,
                                                     round_mode='ceilceil') - object_index_t1_ex)
        object_index_t0_2_t1_1 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t0_ex, flow_t1_ex, relative=False,
                                                     round_mode='ceilfloor') - object_index_t1_ex)
        object_index_t0_2_t1_2 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t0_ex, flow_t1_ex, relative=False,
                                                     round_mode='floorceil') - object_index_t1_ex)
        object_index_t0_2_t1_3 = tf.abs(
            nearest_neighbour_sampler_2d_absolute(object_index_t0_ex, flow_t1_ex, relative=False,
                                                     round_mode='floorfloor') - object_index_t1_ex)

        object_index_t1_2_t0_min = tf.minimum(tf.minimum(object_index_t1_2_t0_0, object_index_t1_2_t0_1),
                                              tf.minimum(object_index_t1_2_t0_2, object_index_t1_2_t0_3))
        object_index_t0_2_t1_min = tf.minimum(tf.minimum(object_index_t0_2_t1_0, object_index_t0_2_t1_1),
                                              tf.minimum(object_index_t0_2_t1_2, object_index_t0_2_t1_3))

        object_t1_2_t0_good_or_boundary = tf.squeeze(tf.less_equal(object_index_t1_2_t0_min, 1E-1), axis=3)
        object_t0_2_t1_good_or_boundary = tf.squeeze(tf.less_equal(object_index_t0_2_t1_min, 1E-1), axis=3)

        # COMPUTE ABSOLUTE FLOW VALUE
        flow_t0_ex_lenght = tf.sqrt(tf.reduce_sum(flow_t0_ex ** 2, axis=3))
        flow_t1_ex_lenght = tf.sqrt(tf.reduce_sum(flow_t1_ex ** 2, axis=3))

        # HERE IS PLUS INSTEAD MINUS BECAUSE FLOW SHOULD BE INVERTED BEFORE SUM
        flow_t0_ex_epe = tf.sqrt(tf.reduce_sum((flow_t0_ex + flow_t1_2_t0) ** 2, axis=3))
        flow_t1_ex_epe = tf.sqrt(tf.reduce_sum((flow_t1_ex + flow_t0_2_t1) ** 2, axis=3))

        object_t0_ex_error = tf.squeeze(tf.abs(object_index_t0_ex - object_index_t1_2_t0), axis=3)
        object_t1_ex_error = tf.squeeze(tf.abs(object_index_t1_ex - object_index_t0_2_t1), axis=3)

        object_t0_ex_dont_care = tf.cast(
            tf.logical_and(object_t1_2_t0_good_or_boundary, tf.greater(object_t0_ex_error, 1E-1)), dtype=tf.float32)
        object_t1_ex_dont_care = tf.cast(
            tf.logical_and(object_t0_2_t1_good_or_boundary, tf.greater(object_t1_ex_error, 1E-1)), dtype=tf.float32)

        const = 3.0
        flow_t0_loss = ((const * flow_t0_ex_epe) / (flow_t0_ex_lenght + 0.01) + object_t0_ex_error)
        flow_t1_loss = ((const * flow_t1_ex_epe) / (flow_t1_ex_lenght + 0.01) + object_t1_ex_error)

        flow_t0_loss = (255.0 * tf.clip_by_value(flow_t0_loss, 0.0, 1.0) * (
                    1.0 - object_t0_ex_dont_care)) + 127 * object_t0_ex_dont_care
        flow_t1_loss = (255.0 * tf.clip_by_value(flow_t1_loss, 0.0, 1.0) * (
                    1.0 - object_t1_ex_dont_care)) + 127 * object_t1_ex_dont_care

        flow_t0_loss = tf.squeeze(flow_t0_loss, axis=0)
        flow_t1_loss = tf.squeeze(flow_t1_loss, axis=0)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        with open('./outfile_flow.txt', 'w') as outfile:
            for j in range(len(filelists)):
                for i in range(len(filelists[j]) + 1 - n_files):

                    try:
                        k = 0

                        if k != 0:
                            outfile.write(' ')
                        outfile.write('{}'.format(filelists[j][i + k]))

                        outfile.write(' {}'.format(flowlists[j][i + k]))
                        outfile.write(' {}'.format(flowlists_back[j][i + k + 1]))

                        occlfile = flowlists[j][i + k]
                        occlfile = occlfile.replace('.pfm', '.png')
                        occlfile = occlfile.replace('optical_flow', 'optical_flow_occlusion_png')
                        outfile.write(' {}'.format(occlfile))

                        occlfile_back = flowlists_back[j][i + k + 1]
                        occlfile_back = occlfile_back.replace('.pfm', '.png')
                        occlfile_back = occlfile_back.replace('optical_flow', 'optical_flow_occlusion_png')
                        outfile.write(' {}'.format(occlfile_back))
                        outfile.write("\n")

                        root_split = occlfile.split('/')
                        root_split = root_split[:-1]
                        occl_dir = os.path.join('', *root_split)
                        create_dir_if_no_exist(occl_dir)

                        root_split = occlfile_back.split('/')
                        root_split = root_split[:-1]
                        occl_dir_right = os.path.join('', *root_split)
                        create_dir_if_no_exist(occl_dir_right)

                        # READING PFM FILES
                        flow_t0_pfm, _ = readPFM(flowlists[j][i + k])
                        flow_t1_pfm, _ = readPFM(flowlists_back[j][i + k + 1])

                        object_index_t0_pfm, _ = readPFM(
                            filelists[j][i + k].replace('frames_cleanpass', 'object_index').replace('.png', '.pfm'))
                        object_index_t1_pfm, _ = readPFM(
                            filelists[j][i + k + 1].replace('frames_cleanpass', 'object_index').replace('.png', '.pfm'))

                        boundaries_t0_pfm = read_pgm(boundaries_lists[j][i + k])
                        boundaries_t1_pfm = read_pgm(boundaries_lists_back[j][i + k + 1])

                        # COMPUTATION OF OCCLUSSION AREAS
                        feed_dict = {flow_t0: flow_t0_pfm, flow_t1: flow_t1_pfm, object_index_t0: object_index_t0_pfm,
                                     object_index_t1: object_index_t1_pfm}
                        flow_t0_loss_numpy, flow_t1_loss_numpy = sess.run([flow_t0_loss, flow_t1_loss], feed_dict=feed_dict)

                        flow_t0_loss_numpy = flow_t0_loss_numpy.astype(np.uint8)
                        flow_t1_loss_numpy = flow_t1_loss_numpy.astype(np.uint8)

                        # SET MOTION BOUNDARIES AS DO NOT CARE
                        boundaries_t0_mask = (boundaries_t0_pfm > 0) & (flow_t0_loss_numpy != 255)
                        boundaries_t1_mask = (boundaries_t1_pfm > 0) & (flow_t1_loss_numpy != 255)
                        flow_t0_loss_numpy[boundaries_t0_mask] = 127
                        flow_t1_loss_numpy[boundaries_t1_mask] = 127

                        with open(occlfile, 'wb') as f:
                            writer = png.Writer(width=flow_t0_loss_numpy.shape[1], height=flow_t0_loss_numpy.shape[0],
                                                bitdepth=8, greyscale=True)
                            z2list = flow_t0_loss_numpy.tolist()
                            writer.write(f, z2list)

                        with open(occlfile_back, 'wb') as f:
                            writer = png.Writer(width=flow_t1_loss_numpy.shape[1], height=flow_t1_loss_numpy.shape[0],
                                                bitdepth=8, greyscale=True)
                            z2list = flow_t1_loss_numpy.tolist()
                            writer.write(f, z2list)

                        if k != 0:
                            outfile.write(' ')
                        outfile.write('{}'.format(filelists[j][i + k].replace('left', 'right')))
                        outfile.write(
                            ' {}'.format(flowlists[j][i + k].replace('_L.pfm', '_R.pfm').replace('left', 'right')))
                        outfile.write(
                            ' {}'.format(flowlists_back[j][i + k + 1].replace('_L.pfm', '_R.pfm').replace('left', 'right')))

                        occlfile = flowlists[j][i + k].replace('_L.pfm', '_R.pfm').replace('left', 'right')
                        occlfile = occlfile.replace('.pfm', '.png')
                        occlfile = occlfile.replace('optical_flow', 'optical_flow_occlusion_png')
                        outfile.write(' {}'.format(occlfile))

                        occlfile_back = flowlists_back[j][i + k + 1].replace('_L.pfm', '_R.pfm').replace('left', 'right')
                        occlfile_back = occlfile_back.replace('.pfm', '.png')
                        occlfile_back = occlfile_back.replace('optical_flow', 'optical_flow_occlusion_png')
                        outfile.write(' {}'.format(occlfile_back))

                        outfile.write("\n")

                        root_split = occlfile.split('/')
                        root_split = root_split[:-1]
                        occl_dir = os.path.join('', *root_split)
                        create_dir_if_no_exist(occl_dir)

                        root_split = occlfile_back.split('/')
                        root_split = root_split[:-1]
                        occl_dir_right = os.path.join('', *root_split)
                        create_dir_if_no_exist(occl_dir_right)

                        flow_t0_pfm, _ = readPFM(
                            flowlists[j][i + k].replace('_L.pfm', '_R.pfm').replace('left', 'right'))
                        flow_t1_pfm, _ = readPFM(
                            flowlists_back[j][i + k + 1].replace('_L.pfm', '_R.pfm').replace('left', 'right'))

                        object_index_t0_pfm, _ = readPFM(
                            filelists[j][i + k].replace('left', 'right').replace('frames_cleanpass',
                                                                                 'object_index').replace('.png', '.pfm'))
                        object_index_t1_pfm, _ = readPFM(
                            filelists[j][i + k + 1].replace('left', 'right').replace('frames_cleanpass',
                                                                                     'object_index').replace('.png',
                                                                                                             '.pfm'))

                        boundaries_t0_pfm = read_pgm(boundaries_lists[j][i + k].replace('left', 'right'))
                        boundaries_t1_pfm = read_pgm(boundaries_lists_back[j][i + k + 1].replace('left', 'right'))

                        feed_dict = {flow_t0: flow_t0_pfm, flow_t1: flow_t1_pfm, object_index_t0: object_index_t0_pfm,
                                     object_index_t1: object_index_t1_pfm}
                        flow_t0_loss_numpy, flow_t1_loss_numpy = sess.run([flow_t0_loss, flow_t1_loss], feed_dict=feed_dict)

                        flow_t0_loss_numpy = flow_t0_loss_numpy.astype(np.uint8)
                        flow_t1_loss_numpy = flow_t1_loss_numpy.astype(np.uint8)

                        # SET MOTION BOUNDARIES AS DO NOT CARE
                        boundaries_t0_mask = (boundaries_t0_pfm > 0) & (flow_t0_loss_numpy != 255)
                        boundaries_t1_mask = (boundaries_t1_pfm > 0) & (flow_t1_loss_numpy != 255)
                        flow_t0_loss_numpy[boundaries_t0_mask] = 127
                        flow_t1_loss_numpy[boundaries_t1_mask] = 127

                        with open(occlfile, 'wb') as f:
                            writer = png.Writer(width=flow_t0_loss_numpy.shape[1], height=flow_t0_loss_numpy.shape[0],
                                                bitdepth=8, greyscale=True)
                            z2list = flow_t0_loss_numpy.tolist()
                            writer.write(f, z2list)

                        with open(occlfile_back, 'wb') as f:
                            writer = png.Writer(width=flow_t1_loss_numpy.shape[1], height=flow_t1_loss_numpy.shape[0],
                                                bitdepth=8, greyscale=True)
                            z2list = flow_t1_loss_numpy.tolist()
                            writer.write(f, z2list)
                    except KeyboardInterrupt:
                        print('End')
                        exit(0)
                    # except:
                    #     print('Error for file: ' + filelists[j][i])
                    #     pass


if __name__ == '__main__':
    main()