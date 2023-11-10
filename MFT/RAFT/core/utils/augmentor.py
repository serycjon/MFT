import contextlib
import numpy as np
from PIL import Image
from MFT.RAFT.core.utils.frame_utils import read_gen
import os
import logging
logger = logging.getLogger(__name__)

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
try:
    import imgaug.augmenters as iaa
    import blend_modes
    from perlin_numpy import generate_perlin_noise_2d
except:
    logger.debug('CANNOT LOAD PERLIN NUMPY')
import string
import random
import copy
import einops

from torchvision.transforms import ColorJitter


class BlendAugmenter:
    def __init__(self, source_dir=None, do_blend_transform=True, blend_prop=0.5, **kwargs):
        if source_dir is None:
            source_dir = "/datagrid/public_datasets/COCO/train2017"
        if do_blend_transform:
            self.image_list = self.list_dir_recursive(source_dir)
        self.do_blend_transform = do_blend_transform
        self.blend_prop = blend_prop
        self.blend_clip_min = kwargs.get('blend_clip_min', 0.5)
        self.blend_clip_max = kwargs.get('blend_clip_max', 0.8)
        self.octaves = kwargs.get('octaves', 8)

    def __call__(self, img1, img2, *args, **kwargs):
        if self.do_blend_transform and np.random.rand() < self.blend_prop:
            img1 = self.rgb2rgba(img1)
            img2 = self.rgb2rgba(img2)
            blend_img = self.generate_blend_image((img1.shape[1], img1.shape[0]))
            blend_img = self.add_perlin_noise_alpha(blend_img, self.blend_clip_min, self.blend_clip_max, self.octaves)
            r = random.uniform(0.0, 0.6)
            img1 = blend_modes.lighten_only(img1, blend_img, r)[:,:,:3]
            img2 = blend_modes.lighten_only(img2, blend_img, r)[:,:,:3]
            img1 = np.round(img1).astype(np.uint8)
            img2 = np.round(img2).astype(np.uint8)
        return img1, img2

    def add_perlin_noise_alpha(self, img, blend_clip_min, blend_clip_max, octaves):
        H, W, _ = img.shape
        perlin_octaves = octaves

        W_b = ((W // perlin_octaves ** 2) + 1) * perlin_octaves ** 2
        H_b = ((H // perlin_octaves ** 2) + 1) * perlin_octaves ** 2

        noise = generate_perlin_noise_2d((H_b, W_b), (perlin_octaves, perlin_octaves))
        noise = noise[:H, :W]

        noise_th = noise - np.min(noise)

        noise_th[noise_th < blend_clip_min] = blend_clip_min
        noise_th[noise_th > blend_clip_max] = blend_clip_max

        noise_th = (noise_th - blend_clip_min)
        noise_th = noise_th / np.max(noise_th)

        img[:, :, 3] = img[:, :, 3] * noise_th
        return img

    def generate_blend_image(self, shape):
        path = np.random.choice(self.image_list)
        blend_img = np.asarray(read_gen(path)).astype(np.float32)
        if blend_img.ndim == 2:
            blend_img = np.stack([blend_img, blend_img, blend_img], axis=2)
        elif blend_img.shape[2] == 1:
            blend_img = np.concatenate([blend_img, blend_img, blend_img], axis=2)
        resized = cv2.resize(blend_img, shape, interpolation=cv2.INTER_AREA)
        return self.rgb2rgba(resized)

    def rgb2rgba(self, img):
        img = img.astype(np.float32)
        ones = np.ones([img.shape[0], img.shape[1], 1], dtype=np.float32)

        return np.concatenate([img, 255 * ones], axis=2)

    def list_dir_recursive(self, path):
        images_list = [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        subdir_list = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        for s in subdir_list:
            images_list += self.list_dir_recursive(s)
        return images_list


class TextAugmenter():
    def __init__(self, do_add_text, max_add_text, add_text_prop, **kwargs):
        self.do_add_text = do_add_text
        self.max_add_text = max_add_text
        self.add_text_prop = add_text_prop
        self.set_text_flow_invalid = kwargs.get('set_text_flow_invalid', True)

        self.max_lenght_text = kwargs.get('max_lenght_text', 20)
        self.min_lenght_text = kwargs.get('min_lenght_text', 5)

        self.font_thickness_max = kwargs.get('font_thickness_max', 5)
        self.font_size_max = kwargs.get('font_size_max', 2.5)

        self.wb_text_prop = kwargs.get('wb_text_prop', 0.5)
        self.alpha_text_prop = kwargs.get('alpha_text_prop', 0.5)

    def __call__(self, img1, img2, valid, *args, **kwargs):
        if self.do_add_text and np.random.rand() < self.add_text_prop:
            layers = np.random.randint(1, self.max_add_text + 1)
            for i in range(layers):
                img1, img2, valid = self.add_text(img1, img2, valid)
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
            valid = valid.astype(np.int32)
        return img1, img2, valid

    def random_string(self, length):
        chars = string.digits + string.ascii_letters + '    '
        result_str = ''.join(random.choice(chars) for i in range(length))
        return result_str

    def add_text(self, img1, img2, valid):
        font = np.random.randint(0, 8)

        if np.random.rand() > self.wb_text_prop:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        else:
            color = np.random.randint(0, 256)
            color = (color, color, color)

        text = self.random_string(np.random.randint(self.min_lenght_text, self.max_lenght_text + 1))
        x_pos = np.random.randint(0, img1.shape[1])
        y_pos = np.random.randint(0, img1.shape[0])
        font_size = np.random.rand() * self.font_size_max
        font_thickness = np.random.randint(1, self.font_thickness_max + 1)

        img1_text = cv2.putText(copy.deepcopy(img1), text, (x_pos, y_pos), font, font_size, color, thickness=font_thickness)
        img2_text = cv2.putText(copy.deepcopy(img2), text, (x_pos, y_pos), font, font_size, color, thickness=font_thickness)

        valid_mask = np.ones_like(img1)
        valid_mask = cv2.putText(valid_mask, text, (x_pos, y_pos), font, font_size, (0, 0, 0), thickness=font_thickness)
        if self.set_text_flow_invalid:
            valid = (valid * (valid_mask[:, :, 0] > 0))

        if np.random.rand() < self.alpha_text_prop:
            alpha = np.random.rand()
            img1_text = alpha * img1_text + (1 - alpha) * img1
            img2_text = alpha * img2_text + (1 - alpha) * img2
        return img1_text, img2_text, valid


class FlowAugmenter:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, **kwargs):

        # spatial augmentation params
        self.load_occlusion = kwargs.get('load_occlusion', False)
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = kwargs.get('spatial_aug_prob', 0.8)
        self.stretch_prob = kwargs.get('stretch_prob', 0.8)
        self.max_stretch = 0.2

        # jpeg transform
        self.do_jpeg_transform = kwargs.get('do_jpeg_transform', False)
        self.jpeg_prop = kwargs.get('jpeg_prop', 0.8)

        # blend transform
        self.blend_source = kwargs.get('blend_source', None)
        self.do_blend_transform = kwargs.get('do_blend_transform', self.blend_source is not None)
        self.blend_prop = 0.5
        self.blend_aug = BlendAugmenter(source_dir=self.blend_source, blend_prop=self.blend_prop, do_blend_transform=self.do_blend_transform)

        # additional text transform
        self.do_add_text_transform = kwargs.get('do_add_text_transform', False)
        self.add_text_prop = kwargs.get('add_text_prop', 0.5)
        self.max_add_text = kwargs.get('max_add_text', 3)
        self.add_text_aug = TextAugmenter(do_add_text=self.do_add_text_transform, max_add_text=self.max_add_text, add_text_prop=self.add_text_prop)

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymetric_photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.asymmetric_color_aug_prob = kwargs.get('asymmetric_color_aug_prob', 1)
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # symmetric
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.asymetric_photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.asymetric_photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        return img1, img2


    def jpeg_transform(self, img1, img2):
        """ JPEG augmentation """

        if self.do_jpeg_transform and np.random.rand() < self.jpeg_prop:
            aug = iaa.imgcorruptlike.JpegCompression(severity=np.random.randint(1,4))
            img1, img2 = aug(images=[img1, img2])
        return img1, img2


    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, occl, valid):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            occl = cv2.resize(occl, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            valid= cv2.resize(valid, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            if len(occl.shape) == 2:
                occl = np.expand_dims(occl, axis=2)
            if len(valid.shape) == 2:
                valid = np.expand_dims(valid, axis=2)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                occl = occl[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:,::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                occl = occl[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]].copy()
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]].copy()
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]].copy()
        occl = occl[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]].copy()
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]].copy()

        return img1, img2, flow, occl, valid

    def out_of_frame_occlusion(self, flow, occl):
        occl_max_value = 1.0
        H,W,C = flow.shape
        x0, y0 = np.meshgrid(np.arange(W), np.arange(H))
        pos_x = flow[:,:,0] + x0
        pos_y = flow[:,:,1] + y0
        occl[pos_x < 0] = occl_max_value
        occl[pos_y < 0] = occl_max_value
        occl[pos_x >= W] = occl_max_value
        occl[pos_y >= H] = occl_max_value
        return occl

    def __call__(self, img1, img2, flow, valid, occl=None, seed=None):
        with tmp_np_seed(seed):
            img1, img2 = self.color_transform(img1, img2)
            if self.load_occlusion is None or not self.load_occlusion:
                img1, img2 = self.eraser_transform(img1, img2)
            if valid is None:
                valid = (np.abs(flow[:,:,0]) < 1000) & (np.abs(flow[:,:,1]) < 1000)
                valid = np.expand_dims(valid, axis=2).astype(float)
            img1, img2, flow, occl, valid = self.spatial_transform(img1, img2, flow, occl, valid)
            img1, img2 = self.blend_aug(img1, img2)
            img1, img2, valid = self.add_text_aug(img1, img2, valid)
            img1, img2 = self.jpeg_transform(img1, img2)

            occl = self.out_of_frame_occlusion(flow, occl)

            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)
            flow = np.ascontiguousarray(flow)
            valid = np.ascontiguousarray(valid)
            occl = np.ascontiguousarray(occl)

            return img1, img2, flow, valid, occl

class SparseFlowAugmenter:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, **kwargs):
        # spatial augmentation params
        self.load_occlusion = kwargs.get('load_occlusion', False)
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # jpeg transform
        self.do_jpeg_transform = kwargs.get('do_jpeg_transform', False)
        self.jpeg_prop = kwargs.get('jpeg_prop', 0.8)

        # blend transform
        self.blend_source = kwargs.get('blend_source', None)
        self.do_blend_transform = kwargs.get('do_blend_transform', self.blend_source is not None)
        self.blend_prop = 0.5
        self.blend_aug = BlendAugmenter(source_dir=self.blend_source, blend_prop=self.blend_prop, do_blend_transform=self.do_blend_transform)

        # additional text transform
        self.do_add_text_transform = kwargs.get('do_add_text_transform', False)
        self.add_text_prop = kwargs.get('add_text_prop', 0.5)
        self.max_add_text = kwargs.get('max_add_text', 3)
        self.add_text_aug = TextAugmenter(do_add_text=self.do_add_text_transform, max_add_text=self.max_add_text, add_text_prop=self.add_text_prop)

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2


    def jpeg_transform(self, img1, img2):
        """ JPEG augmentation """

        if self.do_jpeg_transform and np.random.rand() < self.jpeg_prop:
            aug = iaa.imgcorruptlike.JpegCompression(severity=np.random.randint(1, 4))
            img1, img2 = aug(images=[img1, img2])
        return img1, img2


    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, occl, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            occl = cv2.resize(occl, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            occl = einops.rearrange(occl, 'H W -> H W 1')
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            valid = einops.rearrange(valid, 'H W -> H W 1')

        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                occl = occl[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        occl = occl[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, occl, valid

    def out_of_frame_occlusion(self, flow, occl):
        H,W,C = flow.shape
        x0, y0 = np.meshgrid(np.arange(W), np.arange(H))
        pos_x = flow[:,:,0] + x0
        pos_y = flow[:,:,1] + y0
        occl[pos_x < 0] = 1.0
        occl[pos_y < 0] = 1.0
        occl[pos_x >= W] = 1.0
        occl[pos_y >= H] = 1.0
        return occl

    def __call__(self, img1, img2, flow, valid, occl=None):
        img1, img2 = self.color_transform(img1, img2)
        if self.load_occlusion is None or not self.load_occlusion:
            img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, occl, valid = self.spatial_transform(img1, img2, flow, occl, valid)
        img1, img2 = self.blend_aug(img1, img2)
        img1, img2, valid = self.add_text_aug(img1, img2, valid)
        img1, img2 = self.jpeg_transform(img1, img2)

        occl = self.out_of_frame_occlusion(flow, occl)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        occl = np.ascontiguousarray(occl)

        return img1, img2, flow, valid, occl

@contextlib.contextmanager
def tmp_np_seed(seed):
    if seed is None:
        yield  # do nothing
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
