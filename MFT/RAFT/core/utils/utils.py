import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import einops

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel' or mode == 'viper':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class InputCropAndResize:
    def __init__(self, scale):
        self.scale = scale

    def crop(self, *inputs, **kwargs):
        outputs = []
        for c_input in inputs:
            self.orig_shape = einops.parse_shape(c_input, "N C H W")
            self.new_shape = self.orig_shape.copy()
            self.new_shape['W'] = int(self.orig_shape['W'] * self.scale)
            self.new_shape['H'] = int(self.orig_shape['H'] * self.scale)
            s_w = int((self.orig_shape['W'] - self.new_shape['W']) * self.scale)
            s_h = int((self.orig_shape['H'] - self.new_shape['H']) * self.scale)
            data_cropped = c_input[:, :, s_h:(s_h+self.new_shape['H']), s_w:(s_w+self.new_shape['W'])]
            outputs.append(data_cropped)
        return outputs

    def upsize(self, *inputs, mode='bilinear'):
        size = (self.orig_shape['H'], self.orig_shape['W'])
        return [torch.nn.functional.interpolate(c_input, size, align_corners=True, mode=mode) for c_input in inputs]

    def downsize(self, *inputs, mode='bilinear'):
        size = (self.new_shape['H'], self.new_shape['W'])
        return [torch.nn.functional.interpolate(c_input, size, align_corners=True, mode=mode) for c_input in inputs]

    def downsize_flow(self, *inputs, mode='bilinear'):
        size = (self.new_shape['H'], self.new_shape['W'])

        output_list = []
        for c_input in inputs:
            input_shape = einops.parse_shape(c_input, "N C H W")
            H_ratio = float(self.new_shape['H']) / float(input_shape['H'])
            W_ratio = float(self.new_shape['W']) / float(input_shape['W'])
            flow = torch.nn.functional.interpolate(c_input, size, align_corners=True, mode=mode)
            flow_x = flow[:,0:1,:,:] * W_ratio
            flow_y = flow[:,1:2,:,:] * H_ratio
            output_list.append(torch.cat([flow_x, flow_y], dim=1))
        return output_list



def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def upsample8(maps_data, mode='bilinear'):
    new_size = (8 * maps_data.shape[2], 8 * maps_data.shape[3])
    return F.interpolate(maps_data, size=new_size, mode=mode, align_corners=True)
