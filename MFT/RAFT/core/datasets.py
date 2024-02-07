# -*- origami-fold-style: triple-braces -*-
# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import copy

import numpy as np
import torch
import torch.utils.data as data
import logging

import os
import random
from glob import glob
import os.path as osp

from MFT.RAFT.core.utils import frame_utils
from MFT.RAFT.core.utils.augmentor import FlowAugmenter, SparseFlowAugmenter
from copy import deepcopy

import pickle as pk
from pathlib import Path
import einops
import cv2
from ipdb import iex


# flow is encoded with sign, so 2**15, occlusion and uncertainty without sign, so 2**16:
FLOWOU_IO_FLOW_MULTIPLIER = 2**5  # max-abs-val = 2**(15-5) = 1024, step = 2**(-5) = 0.03
FLOWOU_IO_OCCLUSION_MULTIPLIER = 2**15  # max-val = 2**(16-15) = 2, step = 2**(-15) = 3e-5
FLOWOU_IO_UNCERTAINTY_MULTIPLIER = 2**9   # max-val = 2**(16-9) = 128, step = 2**(-9) = 0.0019


def read_flowou_png(path):
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


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, load_occlusion=False, root=None):
        self.root = root
        self.augmentor = None
        self.sparse = sparse
        self.load_occlusion = load_occlusion
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmenter(**aug_params, load_occlusion=load_occlusion)
            else:
                self.augmentor = FlowAugmenter(**aug_params, load_occlusion=load_occlusion)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.occlusion_list = []
        self.image_list = []
        self.extra_info = []
        self.num_repetitions = 1

        self.logger = logging.getLogger(f'{self.__class__.__name__}')

    def get_reference_frame_path(self, index, relative=False):
        cpath = self.image_list[index][0]
        if relative:
            cpath = cpath.replace(self.root, '')
        return cpath

    def normalise_occlusions_01(self, occl):
        if occl.max() >= 1.1:
            return occl / 255.0
        else:
            return occl

    @iex
    def __getitem__(self, index):
        """
        returns:
            img1: (3, H, W) float32 tensor with 0-255 RGB(!) values
            img2: (3, H, W) float32 tensor with 0-255 RGB(!) values
            flow: (2, H, W) float32 tensor with (xy-ordered?) flow
            valid: (1, H, W) float32 tensor with values 0 (invalid), and 1 (valid)
            occl: (1, H, W) float32 tensor with 0-1 occlusion mask
        """
        index = index % len(self.image_list)

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = einops.rearrange(torch.from_numpy(img1), 'H W C -> C H W', C=2).float()
            img2 = einops.rearrange(torch.from_numpy(img2), 'H W C -> C H W', C=2).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(np.random.randint(0, 1024) + worker_info.id)
                np.random.seed(np.random.randint(0, 1024) + worker_info.id)
                random.seed(np.random.randint(0, 1024) + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.read_gen_sparse_flow(self.flow_list[index])
            valid = einops.rearrange(valid, 'H W -> H W 1')  # np.expand_dims(valid, 2)
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = einops.repeat(img1, 'H W -> H W C', C=3)
            img2 = einops.repeat(img2, 'H W -> H W C', C=3)
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.load_occlusion:
            occl = frame_utils.read_gen(self.occlusion_list[index])
            occl = np.array(occl).astype(np.float32)
            occl = self.normalise_occlusions_01(occl)
        else:
            H, W, C = img1.shape
            occl = np.zeros([H, W, 1], dtype=np.float32)

        if len(occl.shape) == 2:
            occl = einops.rearrange(occl, 'H W -> H W 1')  # occl = np.expand_dims(occl, axis=2)
        else:
            occl = occl[:, :, 0:1]

        if self.augmentor is not None:
            # if self.sparse:
            #     img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            # else:
            #     img1, img2, flow = self.augmentor(img1, img2, flow)
            # orig_occl = occl.copy()
            img1, img2, flow, valid, occl = self.augmentor(img1, img2, flow, valid, occl)

        img1 = einops.rearrange(torch.from_numpy(img1), 'H W C -> C H W', C=3).float()
        img2 = einops.rearrange(torch.from_numpy(img2), 'H W C -> C H W', C=3).float()
        flow = einops.rearrange(torch.from_numpy(flow), 'H W xy -> xy H W', xy=2).float()
        occl = einops.rearrange(torch.from_numpy(occl), 'H W 1 -> 1 H W').float()

        if valid is not None:
            valid = einops.rearrange(torch.from_numpy(valid), 'H W 1 -> 1 H W') > 0.99
            valid = valid & einops.rearrange(torch.all(flow.abs() < 1000, dim=0), 'H W -> 1 H W')
        else:
            valid = einops.rearrange(torch.all(flow.abs() < 1000, dim=0), 'H W -> 1 H W')

        return img1, img2, flow, valid.float(), occl

    def __rmul__(self, v):
        assert isinstance(v, int)
        self.num_repetitions *= v
        return self

    def __len__(self):
        return len(self.image_list) * self.num_repetitions

    def load_cache(self, file_path):
        self.logger.info("Loading cache")
        file_path = f'{file_path}.pkl'
        if not os.path.isfile(file_path):
            return False
        with open(file_path, 'rb') as f:
            # files = np.load(f, allow_pickle=True)
            files = pk.load(f)

        self.image_list = files.get('image_list')
        self.flow_list = files.get('flow_list')
        self.occlusion_list = files.get('occlusion_list')
        self.multi_flow_list = files.get('multi_flow_list')
        self.multi_image_list = files.get('multi_image_list')
        self.multi_occl_list = files.get('multi_occl_list')
        self.extra_info = files.get('extra_info')
        self.flow_zero_list = files.get('flow_zero_list')
        self.logger.info("Done loading cache")
        return True

    def save_cache(self, file_path, additional_files=None):
        file_path = f'{file_path}.pkl'
        files = {'image_list': self.image_list,
                 'flow_list': self.flow_list,
                 'occlusion_list': self.occlusion_list,
                 'extra_info': self.extra_info}
        if additional_files is not None:
            files.update(additional_files)
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pk.dump(files, f)
            # np.save(f, files, allow_pickle=True)

    @staticmethod
    def bw_bilinear_interpolate_flow_numpy(im, flow):

        def _bw_bilin_interp(im, x, y):
            x = np.asarray(x)
            y = np.asarray(y)

            x0 = np.floor(x).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y).astype(int)
            y1 = y0 + 1

            x0 = np.clip(x0, 0, im.shape[1] - 1)
            x1 = np.clip(x1, 0, im.shape[1] - 1)
            y0 = np.clip(y0, 0, im.shape[0] - 1)
            y1 = np.clip(y1, 0, im.shape[0] - 1)

            Ia = im[y0, x0]
            Ib = im[y1, x0]
            Ic = im[y0, x1]
            Id = im[y1, x1]

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        ndim = im.ndim
        if ndim == 2:
            im = np.expand_dims(im, axis=2)
        H, W, C = im.shape
        X_g, Y_g = np.meshgrid(range(W), range(H))
        x, y = flow[:, :, 0], flow[:, :, 1]
        x = x + X_g
        y = y + Y_g
        im_w = []
        for i in range(C):
            im_w.append(_bw_bilin_interp(im[:, :, i], x, y))
        im_w = np.stack(im_w, axis=2)

        if ndim == 2:
            im_w = im_w[:, :, 0]
        return im_w


class KubricDataset(FlowDataset):
    def __init__(self, aug_params=None, split='train',
                 root='datasets/kubric_movi_e_longterm', load_occlusion=False,
                 upsample2=False, correct_flow=False):
        """
        """
        super(KubricDataset, self).__init__(aug_params, load_occlusion=load_occlusion, root=root)
        self.flow_zero_list = []
        self.multi_flow_list = []
        self.multi_image_list = []
        self.upsample2 = upsample2
        self.correct_flow = correct_flow

        if split == 'test':
            self.is_test = True

        self.save_file_path = f'train_files_lists/Kubric_Pixel_Tracking_{split}'

        if not self.load_cache(self.save_file_path):
            data_root = osp.join(root, split)

            for idx, scene in enumerate(os.listdir(data_root)):
                # if idx >= 9:
                #     break

                image_list = sorted(glob(osp.join(data_root, scene, 'images', '*.png')))
                flow_list = sorted(glob(osp.join(data_root, scene, 'flowou', '*.flowou.png')))

                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[0], image_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        # +1 because of flow from 0 to 0 (first flow is saved only to see problems with discretisation)
                        self.flow_list += [flow_list[i+1]]
                        self.flow_zero_list += [flow_list[0]]

                self.multi_image_list.append(image_list)
                self.multi_flow_list.append(flow_list)

            self.save_cache(self.save_file_path, {'extra_info': self.extra_info,
                                                  'multi_image_list': self.multi_image_list,
                                                  'multi_flow_list': self.multi_flow_list,
                                                  'flow_zero_list': self.flow_zero_list})

    def get_data_delta(self, index, delta=None):
        if delta is None:
            im1_path = self.image_list[index][0]
            im2_path = self.image_list[index][1]
        else:
            im1_path = self.multi_image_list[index][0]
            im2_path = self.multi_image_list[index][delta]

        if self.is_test:
            img1 = frame_utils.read_gen(im1_path)
            img2 = frame_utils.read_gen(im2_path)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if delta is None:
            flowou_path = self.flow_list[index]
            flowou_zero_path = self.flow_zero_list[index]
        else:
            flowou_path = self.multi_flow_list[index][delta]
            flowou_zero_path = self.multi_flow_list[index][0]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow, occlusions, _ = read_flowou_png(flowou_path)
        flow = einops.rearrange(flow, 'C H W -> H W C', C=2).astype(np.float32)
        occl = einops.rearrange(occlusions, 'C H W -> H W C', C=1).astype(np.float32)
        occl = self.normalise_occlusions_01(occl)

        if self.correct_flow and delta != 0:
            flow_zero, _, _ = read_flowou_png(flowou_zero_path)
            flow_zero = einops.rearrange(flow_zero, 'C H W -> H W C', C=2).astype(np.float32)
            obj_mask_bin = flow_zero[:, :, 0] > 0.25  # should be exactly 0.0 or 0.5, but there is error due to saving to discrete flowou  # noqa E501
            obj_mask_float = obj_mask_bin.astype(np.float32) - 0.5
            flow_zero[np.logical_not(obj_mask_bin)] = 0.
            flow_zero[obj_mask_bin] = 0.5

            flow = flow - flow_zero
            flow = self.bw_bilinear_interpolate_flow_numpy(flow, -flow_zero)
            obj_mask_float = self.bw_bilinear_interpolate_flow_numpy(obj_mask_float, -flow_zero) + 0.5
            occl = self.bw_bilinear_interpolate_flow_numpy(occl, -flow_zero)
            valid = np.logical_or(obj_mask_float > 0.99, obj_mask_float < 0.01)
            valid = np.expand_dims(valid, axis=2).astype(float)

        img1 = frame_utils.read_gen(im1_path)
        img2 = frame_utils.read_gen(im2_path)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.augmentor is not None:
            img1, img2, flow, valid, occl = self.augmentor(img1, img2, flow, valid, occl)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        occl = torch.from_numpy(occl).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).permute(2, 0, 1).float() > 0.99
            valid = valid & torch.unsqueeze((flow[0].abs() < 1000) & (flow[1].abs() < 1000), dim=0)
        else:
            valid = torch.unsqueeze((flow[0].abs() < 1000) & (flow[1].abs() < 1000), dim=0)

        return img1, img2, flow, valid.float(), occl

    def __getitem__(self, index):
        return self.get_data_delta(index)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel-complete',
                 dstype='clean', load_occlusion=False, subsplit=None):
        """
        :param subsplit: None : whole training dataset
                         'validation' : self.validation_subplit_dirs sequences only
                         'training' : all sequences except validation sequences
        """
        super(MpiSintel, self).__init__(aug_params, load_occlusion=load_occlusion, root=root)
        self.logger = logging.getLogger(f'{self.__class__.__name__}-{dstype}')

        if split == 'test':
            self.is_test = True
        self.validation_subsplit_dirs = ['alley_1', 'ambush_6', 'bamboo_2', 'cave_4', 'market_5', 'shaman_3']

        if subsplit is not None:
            self.save_file_path = f'train_files_lists/MpiSintel_{split}_{dstype}_{subsplit}'
        else:
            self.save_file_path = f'train_files_lists/MpiSintel_{split}_{dstype}'

        if not self.load_cache(self.save_file_path):
            self.logger.warning(f"Could not load cache from {self.save_file_path}")
            flow_root = osp.join(root, split, 'flow')
            occl_root = osp.join(root, split, 'occlusions_rev')
            image_root = osp.join(root, split, dstype)

            for scene in os.listdir(image_root):

                if subsplit is not None:
                    if subsplit == 'training' and scene in self.validation_subsplit_dirs:
                        continue
                    elif subsplit == 'validation' and scene not in self.validation_subsplit_dirs:
                        continue

                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                    self.occlusion_list += sorted(glob(osp.join(occl_root, scene, '*.png')))

            self.save_cache(self.save_file_path)


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params, root=root)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass', load_occlusion=False):
        super(FlyingThings3D, self).__init__(aug_params, load_occlusion=load_occlusion, root=root)

        self.save_file_path = f'train_files_lists/FlyingThings3D_{dstype}'
        if not self.load_cache(self.save_file_path):
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            occl, im1, im2, flow = None, None, None, None
                            if direction == 'into_future':
                                im1 = images[i]
                                im2 = images[i+1]
                                flow = flows[i]
                                occl = flows[i].replace('optical_flow', 'optical_flow_occlusion_png') \
                                               .replace('.pfm', '.png')
                            elif direction == 'into_past':
                                im1 = images[i + 1]
                                im2 = images[i]
                                flow = flows[i + 1]
                                occl = flows[i + 1].replace('optical_flow', 'optical_flow_occlusion_png') \
                                                   .replace('.pfm', '.png')

                            if all([os.path.isfile(x) for x in [occl, im1, im2, flow]]):
                                self.image_list += [[im1, im2]]
                                self.flow_list += [flow]
                                self.occlusion_list += [occl]

            self.save_cache(self.save_file_path)


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI/basic/'):
        super(KITTI, self).__init__(aug_params, sparse=True, root=root)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        print(len(self.flow_list))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True, root=root)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """
    train_dataset = None
    load_occlusion = args.occlusion_module is not None
    if args.dashcam_augmenentation:
        aug_params = {'do_jpeg_transform': True,
                      'do_blend_transform': False,
                      'do_add_text_transform': False,
                      'jpeg_prop': 0.5,
                      }
    else:
        aug_params = {}

    if args.stage == 'chairs':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True})
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel_things':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion)
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion)
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things_clean + things_final

    elif args.stage == 'sintel_things_train_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        train_dataset = 200 * sintel_clean + 200 * sintel_final + things_final

    elif args.stage == 'sintel_things_kubric_train_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion, subsplit='training')  # noqa 501

        kubric_aug_params = copy.deepcopy(aug_params)
        kubric_aug_params.update({'min_scale': 1.8, 'max_scale': 2.2, 'stretch_prob': 1.1, 'spatial_aug_prob': 1.1,
                                  'asymmetric_color_aug_prob': 0.0})
        kubric_train = KubricDataset(kubric_aug_params, split='train', load_occlusion=load_occlusion, correct_flow=True)
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things_final + kubric_train

    elif args.stage == 'sintel':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True})
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion)

        if TRAIN_DS == 'C+T+K+S+H':
            kitti_aug_params = deepcopy(aug_params)
            kitti_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # noqa 501
            hd1k_aug_params = deepcopy(aug_params)
            hd1k_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            kitti = KITTI(kitti_aug_params)
            hd1k = HD1K(hd1k_aug_params)
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

        elif TRAIN_DS == 'C+T+K+S+H+V':
            kitti_aug_params = deepcopy(aug_params)
            kitti_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # noqa 501
            hd1k_aug_params = deepcopy(aug_params)
            hd1k_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            kitti = KITTI(kitti_aug_params)
            hd1k = HD1K(hd1k_aug_params)
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

    elif args.stage == 'kitti':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False})
        train_dataset = KITTI(aug_params, split='training')

    num_workers = getattr(args, 'n_workers', 8)
    print(f"num_workers: {num_workers}")

    shuffle = not getattr(args, 'no_shuffle', False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def combine_datasets_with_weights(weight_dataset_pairs):
    multipliers = np.array([weight / len(dataset) for weight, dataset in weight_dataset_pairs])
    multipliers /= np.amin(multipliers)
    multipliers = np.round(multipliers).astype(np.int32).tolist()
    print(f"Datasets combined with multipliers: {multipliers}")
    datasets = [dataset for weight, dataset in weight_dataset_pairs]
    lengths = [len(dataset) for dataset in datasets]
    print(f"result in sample counts: {[mult * length for mult, length in zip(multipliers, lengths)]}")
    weighted_datasets = [int(mult) * dataset for mult, dataset in zip(multipliers, datasets)]

    result = weighted_datasets[0]
    for dataset in weighted_datasets[1:]:
        result += dataset

    return result
