import torch
import torch.nn.functional as F
import einops
import logging

from MFT.utils.geom_utils import torch_get_featuremap_coords, get_featuremap_coords
from MFT.utils.misc import ensure_numpy

from MFT.RAFT.core.raft import RAFT  # noqa: E402
from MFT.RAFT.core.utils.utils import InputPadder  # noqa: E402

logger = logging.getLogger(__name__)


class RAFTWrapper():
    def __init__(self, config):
        device = 'cuda'
        self.C = config

        model = torch.nn.DataParallel(RAFT(self.C.raft_params))
        model.load_state_dict(torch.load(self.C.model, map_location='cpu'))

        model = model.module
        model.requires_grad_(False)
        model.to(device)
        model.eval()

        self.model = model

    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, src_img_identifier=None,
                     numpy_out=False, init_flow=None, vis_debug=False):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
            init_flow: (xy H W) flow
        """
        H, W = src_img.shape[:2]
        # the_flow_timer = time_measurer('ms')
        image1 = einops.rearrange(torch.from_numpy(src_img[:, :, ::-1].copy()),
                                  'H W C -> 1 C H W', C=3)
        image2 = einops.rearrange(torch.from_numpy(dst_img[:, :, ::-1].copy()),
                                  'H W C -> 1 C H W', C=3)
        image1, image2 = image1.cuda().float(), image2.cuda().float()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        if init_flow is not None:
            init_flow = init_flow.to('cuda').to(torch.float32)
            init_flow, = padder.pad(einops.rearrange(init_flow, 'xy H W -> 1 xy H W', xy=2))
            init_flow = downsample_flow_8(init_flow)

        all_predictions = self.model(image1, image2, iters=self.C.flow_iters, test_mode=True,
                                     flow_init=init_flow, vis_debug=vis_debug)

        flow_pre = padder.unpad(all_predictions['flow'])
        flow = flow_pre[0]  # xy, H, W
        assert flow.shape == (2, H, W)
        occlusions = torch.squeeze(padder.unpad(all_predictions['occlusion'].softmax(dim=1)[:, 1:2, :, :]), dim=0)
        uncertainty_pred = torch.squeeze(padder.unpad(all_predictions['uncertainty']), dim=0)
        sigma = torch.sqrt(torch.exp(uncertainty_pred))

        if mode == 'flow':
            if numpy_out:
                flow = ensure_numpy(flow)
                occlusions = ensure_numpy(occlusions)
                sigma = ensure_numpy(sigma)

            extra_outputs = {'occlusion': occlusions,
                             'sigma': sigma,
                             'debug': all_predictions.get('debug', None)}
            return flow, extra_outputs

        elif mode == 'TC':
            flow_flat = einops.rearrange(flow, 'delta H W -> delta (H W)', delta=2)
            flow_shape = einops.parse_shape(flow, 'delta H W')
            self.last_flow_shape = flow_shape
            if isinstance(flow_flat, torch.Tensor):
                src_coords = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_flat.device)
            else:
                src_coords = get_featuremap_coords((flow_shape['H'], flow_shape['W']))
            dst_coords = src_coords + flow_flat

            if numpy_out:
                src_coords = ensure_numpy(src_coords)
                dst_coords = ensure_numpy(dst_coords)
                occlusions = ensure_numpy(einops.rearrange(occlusions, '1 H W -> (H W)'))
                sigma = ensure_numpy(einops.rearrange(sigma, '1 H W -> (H W)'))

            extra_outputs = {'occlusion': occlusions,
                             'sigma': sigma,
                             'debug': all_predictions.get('debug', None)}
            return src_coords, dst_coords, extra_outputs



def downsample_flow_8(flow, mode='bilinear'):
    """Downsample a (B, xy, H, W) flow tensor to (B, xy, H/8, W/8) (assume divisible)"""
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8
