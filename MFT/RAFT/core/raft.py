import torch
import torch.nn as nn
import torch.nn.functional as F

from MFT.RAFT.core.update import BasicUpdateBlock, SmallUpdateBlock, OcclusionAndUncertaintyBlock
from MFT.RAFT.core.extractor import BasicEncoder, SmallEncoder
from MFT.RAFT.core.corr import CorrBlock, AlternateCorrBlock
from MFT.RAFT.core.utils.utils import coords_grid, upflow8, upsample8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.occlusion_estimation = args.occlusion_module is not None
        self.uncertainty_estimation = self.occlusion_estimation and 'with_uncertainty' in args.occlusion_module
        self.OU_last_iter_only = getattr(args, 'OU_last_iter_only', False)
        self.relu_uncertainty = getattr(args, 'relu_uncertainty', False)
        if self.uncertainty_estimation:
            self.mult_uncetrainty_upsample = 8.0 if 'upsample8' in args.occlusion_module else 1.0
            self.eps_uncertainty = 10e-4

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.size_occl_uncer_input_dims = 712

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        if args.occlusion_module is not None:
            self.occlusion_block = OcclusionAndUncertaintyBlock(self.args, hidden_dim=self.size_occl_uncer_input_dims)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, mult_coef=8.0, n_channels=2):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(mult_coef * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, n_channels, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, n_channels, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False, normalise_input=True, return_features=False, return_coords=True, vis_debug=False):
        """
        Estimate optical flow, occlusion and uncertainty between pair of frames

        :param image1:     input RGB image - torch.float32 with shape [B,3,H,W],
                           INPUT HAS TO BE PADDED to be divisible by 8
                           input data in range [0,255] (default, normalise_input=True) OR [-1,1] (normalise_input=False)
        :param image2:     same as image1
        :param iters:      number of raft iterations
        :param flow_init:  initial value for flow [B,2,H,W] torch.flow32
        :param upsample:   UNUSED
        :param test_mode:  change output of the RAFT - only final iteration is outputed
        :param normalise_input: enable normalisation of input images [0,255] -> [-1,1]
        :param return_features: DISABLED, return raft features for further processing
        :param return_coords:   return non-upsampled optical flow (1/8 resolution) [B,2,H/8,W/8]
        :param vis_debug: return additional info for visual debugging
        :return: dict with results: (DO NOT FORGET REMOVE PADDING)
            outputs['flow']        - estimated optical flow [B,2,H,W]
            outputs['occlusion']   - estimated occlusion map [B,2,H,W],
                                   - further processing needed occl = outputs['occlusion'].softmax(dim=1)[:,1:2,:,:]
            outputs['uncertainty'] - estimated uncertainty map [B,1,H,W],
                                   - sigma = torch.sqrt(torch.exp(outputs['uncertainty']))
            outputs['coords']      - non-upsampled optical flow (1/8 resolution) [B,2,H/8,W/8]
        """

        if normalise_input:
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius,
                                normalized_features=getattr(self.args, 'normalized_features', False))

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        occl_predictions = []
        uncertainty_predictions = []
        if vis_debug:
            debug_stuff = {
                # the costvolume-pyramid is a list of response maps (the
                # first is the costvolume, the others are created by
                # iterative avgpooling with kernel size 2 and stride 2)
                #
                # the shape is (batch * h1 * w1, dim, h2, w2) with h1, w1
                # being from reference image (H/8, W/8) and h2, w2 from
                # the right image (size depends on pyramid level floor(h / # 2**lvl))
                'costvolume_pyramid': [lvl.detach().cpu() for lvl in corr_fn.corr_pyramid],
                # coordinates with shape (1, xy(2), h1, w1)
                'coords_left': coords0.detach().cpu(),
                'iterations': [],
            }
        for itr in range(iters):
            coords1 = coords1.detach()
            if vis_debug:
                debug_stuff['iterations'].append({'coords': coords1.cpu()})
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, motion_features = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

            # in test time - occlusion and uncertainty estimated only for LAST iteration
            if (test_mode or self.OU_last_iter_only) and (itr != (iters - 1)):
                continue

            if self.occlusion_estimation:
                occlusion, uncertainty = self.occlusion_block(net.detach(),  # hidden GRU state
                                                              inp,  # context features
                                                              corr.detach(),  # correlation cost-volume + pyramid
                                                                              # ^ sampled at the previous flow position
                                                              (coords1 - coords0).detach(),  # flow
                                                              delta_flow.detach(),  # flow delta in last step
                                                              motion_features  # encoded cost-volume sample + flow
                                                              )

                if up_mask is None:
                    occl_up = upsample8(occlusion)  # upsample only
                else:
                    occl_up = self.upsample_flow(occlusion, up_mask, mult_coef=1.0)  # upsample only
                occl_predictions.append(occl_up)

                if self.uncertainty_estimation:
                    if up_mask is None:
                        uncertainty_up = upsample8(uncertainty) * self.mult_uncetrainty_upsample # upsample and multiply by 8 or 1
                    else:
                        uncertainty_up = self.upsample_flow(uncertainty, up_mask, mult_coef=self.mult_uncetrainty_upsample, n_channels=1) # upsample and multiply by 8
                        if self.relu_uncertainty:
                            uncertainty_up = F.relu(uncertainty_up)

                        if getattr(self.args, 'experimental_cleanup', False):
                            uncertainty_up[~torch.isfinite(uncertainty_up)] = 35
                            uncertainty_up[uncertainty_up > 35] = 35
                            # print('cleaning up the mess')
                    uncertainty_predictions.append(uncertainty_up)

        if return_features:
            # raise NotImplementedError('removed back compatibility, last commit with compatible output: bd887365564f9a011987d29c6f518561c2f8b9db')
            context_features = torch.cat([cnet, fmap1], dim=1)
            return flow_up, context_features

        outputs = dict()
        if test_mode:
            outputs['flow'] = flow_up
            if self.uncertainty_estimation:
                outputs['uncertainty'] = uncertainty_up
            if self.occlusion_estimation:
                outputs['occlusion'] = occl_up

        else:
            outputs['flow'] = flow_predictions
            if self.uncertainty_estimation:
                outputs['uncertainty'] = uncertainty_predictions
            if self.occlusion_estimation:
                outputs['occlusion'] = occl_predictions

        if return_features:
            context_features = torch.cat([cnet, fmap1], dim=1)
            outputs['features'] = context_features

        if return_coords:
            outputs['coords'] = coords1 - coords0

        if vis_debug:
            debug_stuff['iterations'].append({'coords': coords1.cpu()}) # after last iteration
            outputs['debug'] = debug_stuff

        return outputs
