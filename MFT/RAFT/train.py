from __future__ import print_function, division
import sys
sys.path.append('core')
from ipdb import iex

import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim


from MFT.RAFT.core.raft import RAFT
from MFT.RAFT import evaluate
import MFT.RAFT.core.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from MFT.RAFT.core.utils.timer import Timer

from MFT.RAFT.core.utils.flow_viz import flow_to_color

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 10 # 100
VAL_FREQ = 5000 # 5000


def sequence_loss(preds_dict, flow_gt, valid, occl_gt=None, gamma=0.8, max_flow=MAX_FLOW, args=None, **kwargs):
    alpha_flow = kwargs.get('alpha_flow', 1.0)
    alpha_occl = kwargs.get('alpha_occl', 5.0)
    alpha_uncertainty = kwargs.get('alpha_uncertainty', 1.0)

    uncertainty_loss_type = args.uncertainty_loss
    weighting_unc_loss = args.weighting_unc_loss
    flow_loss_type = args.optical_flow_loss

    total_loss = 0.0
    metrics = {}

    flow_preds = preds_dict['flow']

    if not args.freeze_optical_flow_training:
        flow_loss, flow_metrics = sequence_flow_loss(flow_preds, flow_gt, valid, occl_gt=occl_gt,
                                                     gamma=gamma, max_flow=max_flow, flow_loss_type=flow_loss_type)
        metrics.update(flow_metrics)
        total_loss += (alpha_flow * flow_loss)

    if args.occlusion_module is not None:
        occl_preds = preds_dict['occlusion']
        occl_loss, occl_metrics = sequence_occl_loss(occl_preds, occl_gt, flow_gt, valid, gamma=gamma, max_flow=max_flow)
        metrics.update(occl_metrics)
        total_loss += (alpha_occl * occl_loss)

    if args.occlusion_module is not None and 'uncertainty' in args.occlusion_module:
        uncertainty_preds = preds_dict['uncertainty']
        uncertainty_loss, uncertainty_metrics = sequence_uncertainty_loss(flow_preds, uncertainty_preds,
                                                                          flow_gt, valid, gamma=gamma,
                                                                          max_flow=max_flow,
                                                                          uncertainty_loss_type=uncertainty_loss_type,
                                                                          weighting_unc_loss=weighting_unc_loss,
                                                                          occl_gt=occl_gt)
        metrics.update(uncertainty_metrics)
        total_loss += (alpha_uncertainty * uncertainty_loss)

    return total_loss, metrics

def sequence_occl_loss(occl_preds, occl_gt, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(occl_preds)
    occl_loss = 0.0

    cross_ent_loss = nn.CrossEntropyLoss(reduction='none')

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)
    # only 100% occluded and 100% non-occluded are used for training
    occl_valid = torch.logical_or(occl_gt < 0.01, occl_gt > 0.99)
    valid = torch.logical_and(occl_valid[:,0,:,:], valid)

    occl_gt_thresholded = occl_gt > 0.5
    occl_gt_thresholded = occl_gt_thresholded[:,0,:,:].long()

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = cross_ent_loss(occl_preds[i].softmax(dim=1), occl_gt_thresholded)
        occl_loss += i_weight * (valid[:, None] * i_loss).mean()

    metrics = {
        'train/cross_entropy_occl': i_loss.mean().item(),
    }

    return occl_loss, metrics


def sequence_flow_loss(flow_preds, flow_gt, valid, occl_gt=None, gamma=0.8, max_flow=MAX_FLOW, flow_loss_type='L1'):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)
    if 'occl' in flow_loss_type:
        assert occl_gt is not None
        hard_occl_mask = torch.squeeze(occl_gt[:,0,:,:], dim=1) > 0.99

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()

        if flow_loss_type == 'L1':
            i_valid = valid
        elif flow_loss_type == 'L1_non_occluded':
            i_valid = torch.logical_and(valid, torch.logical_not(hard_occl_mask))
        elif flow_loss_type == 'L1_occluded_to_epe3':
            flow_epe = torch.sqrt(torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1, keepdim=False)).detach()
            epe_mask = flow_epe < 3.0
            nonoccl_or_epe_mask = torch.logical_or(torch.logical_not(hard_occl_mask), epe_mask)
            i_valid = torch.logical_and(valid, nonoccl_or_epe_mask)
        else:
            raise NotImplementedError(f'Flow loss type {flow_loss_type} not implemented')
        flow_loss += i_weight * (i_valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train/epe': epe.mean().item(),
        'train/1px': (epe < 1).float().mean().item(),
        'train/3px': (epe < 3).float().mean().item(),
        'train/5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def weights_uncertainty_according_epe(epe):
    device = epe.device
    coef = np.array([-7.27864588e-02,  9.00020608e+00, -1.79078330e+01,  8.68281513e+01])
    epe_clamped = torch.clamp(epe, 0, 50).detach()
    epe2 = epe_clamped**2
    epe3 = epe_clamped**3

    weight = epe3 * coef[0] + epe2 * coef[1] + epe_clamped * coef[2] + coef[3]
    weight = weight / 50
    return weight

def sequence_uncertainty_loss(flow_preds, uncertainty_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW,
                              uncertainty_loss_type='huber', weighting_unc_loss=False, occl_gt=None):
    """
    InProceedings (He2019)
    He, Y.; Zhu, C.; Wang, J.; Savvides, M. & Zhang, X.
    Bounding box regression with uncertainty for accurate object detection
    Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2019, 2888-2897

    Loss from equation 9 and 10
    Loss is weighted (i_weight) in the same way as in the RAFT
    """

    if uncertainty_loss_type in ['huber', 'huber_non_occluded']:
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    elif uncertainty_loss_type in ['L2', 'L2_non_occluded']:
        unc_loss = torch.nn.MSELoss(reduction='none')
    elif uncertainty_loss_type == 'huber_epe_direct':
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    elif uncertainty_loss_type == 'huber_epe_direct_non_occluded':
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError('This type of loss is not implemented for uncertainty')

    n_predictions = len(flow_preds)
    uncertainty_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)

    if uncertainty_loss_type in ['huber', 'L2', 'huber_non_occluded', 'L2_non_occluded']:
        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)

            i_alpha = uncertainty_preds[i]
            i_loss_exp_alpha = torch.exp(-i_alpha)
            if uncertainty_loss_type == 'L2':
                i_loss_exp_alpha = 0.5 * i_loss_exp_alpha

            flow_epe = torch.sqrt(torch.sum((flow_preds[i] - flow_gt)**2, dim=1, keepdim=True)).detach()

            unc_loss_comp = unc_loss(flow_epe, torch.zeros_like(flow_epe))
            i_loss = i_loss_exp_alpha * unc_loss_comp + 0.5 * i_alpha

            if 'non_occluded' in uncertainty_loss_type:
                valid = torch.logical_and(valid, torch.logical_not(torch.squeeze(occl_gt, dim=1) > 0.99))
            if weighting_unc_loss:
                i_loss = weights_uncertainty_according_epe(unc_loss_comp) * i_loss
            uncertainty_loss += i_weight * (valid[:, None] * i_loss).mean()
    elif uncertainty_loss_type in ['huber_epe_direct', 'huber_epe_direct_non_occluded']:
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)

            i_alpha = uncertainty_preds[i]
            i_exp_alpha = torch.exp(-i_alpha)
            flow_L2 = torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1, keepdim=True).detach()
            flow_epe = torch.sqrt(flow_L2)

            i_comp_for_alpha = - i_alpha * i_exp_alpha

            i_loss = unc_loss(i_comp_for_alpha, flow_L2)

            if 'non_occluded' in uncertainty_loss_type:
                valid = torch.logical_and(valid, torch.logical_not(torch.squeeze(occl_gt, dim=1) > 0.99))
            if weighting_unc_loss:
                i_loss = weights_uncertainty_according_epe(flow_epe) * i_loss
            uncertainty_loss += i_weight * (valid[:, None] * i_loss).mean()
    else:
        raise NotImplementedError('This type of loss is not implemented for uncertainty')

    metrics = {
        'train/uncert': i_loss.mean().item(),
    }

    return uncertainty_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, logfile_comment=''):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.logfile_comment = logfile_comment

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(comment=self.logfile_comment)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def create_writer_if_not_set(self):
        if self.writer is None:
            self.writer = SummaryWriter(comment=self.logfile_comment)

    def write_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_image_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_image(key, results[key], self.total_steps)

    def write_images(self, inputs):
        self.create_writer_if_not_set()

        for key in inputs:
            im = inputs[key]

            # grid = torchvision.utils.make_grid(im)
            # self.writer.add_image(key, grid, self.total_steps)
            if key == 'valid':
                data = im.type(torch.uint8) * 255
                # data = torch.unsqueeze(data, 1) * 255
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'occl' in key:
                data = torch.clamp(im * 255., 0., 255.)
                data = data.type(torch.uint8)
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'sigma' in key:
                data = torch.clamp(im, 0., 255.)
                data = data.type(torch.uint8)
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'flow' in key:
                data = im.detach().cpu().numpy().transpose(0,2,3,1)
                color_list = []
                for i in range(data.shape[0]):
                    color_list.append(flow_to_color(data[i, :, :, :]))
                    color_image = np.stack(color_list, axis=0)
                self.writer.add_images(key, color_image, dataformats='NHWC', global_step=self.total_steps)
            else:
                self.writer.add_images(key, im.type(torch.uint8), dataformats='NCHW', global_step=self.total_steps)

    def close(self):
        self.writer.close()

def weight_freezer(model, args):
    if args.freeze_optical_flow_training or args.freeze_features_training:
        model.eval()
        model.requires_grad_(False)
        if not args.freeze_optical_flow_training:
            raise NotImplementedError('Have to be specified')
        if not args.freeze_features_training:
            raise NotImplementedError('Have to be specified')
        model.module.occlusion_block.requires_grad_(True)
        model.module.occlusion_block.train()
    else:
        model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    return model



@iex
def train(args):

    train_timer = Timer()

    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(gpu_n) for gpu_n in args.gpus])

    args.gpus = range(len(args.gpus))
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model = weight_freezer(model, args)

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, logfile_comment=args.name)


    should_keep_training = True
    print('Training...', train_timer.iter(), train_timer())
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            # print(i_batch, total_steps, train_timer.iter(), train_timer())

            optimizer.zero_grad()
            image1, image2, flow, valid, occl = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            all_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(all_predictions, flow, valid, occl_gt=occl, gamma=args.gamma, args=args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if (total_steps == 7) or (total_steps % VAL_FREQ == VAL_FREQ - 1):
                print('validation ', i_batch, total_steps, train_timer.iter(), train_timer())
                PATH = args.checkpoints + '/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                # print('before val: ', train_timer.iter(), train_timer())
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'sintel_val_subsplit':
                        results.update(evaluate.validate_sintel(model.module, subsplit='validation'))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                print('after val: ', total_steps, train_timer.iter(), train_timer())

                logger.write_dict(results)

                logger.write_images({'image1': image1, 'image2': image2, 'valid': valid})
                logger.write_images({'flow_gt': flow})
                if occl is not None:
                    logger.write_images({'occl_gt': occl})
                flow_predictions = all_predictions['flow']
                logger.write_images({'flow_est': flow_predictions[-1]})
                if model.module.occlusion_estimation:
                    occl_predictions = all_predictions['occlusion']
                    occl_prediction_single = occl_predictions[-1].softmax(dim=1)
                    logger.write_images({'occl_est_neg': 255. * occl_prediction_single[:,0:1,:,:] })
                    logger.write_images({'occl_est_pos': 255. * occl_prediction_single[:,1:2,:,:] })
                if model.module.uncertainty_estimation:
                    uncertainty_predictions = all_predictions['uncertainty']
                    sigma2 = torch.exp(uncertainty_predictions[-1])
                    logger.write_images({'sigma2_est': sigma2 * 255})
                    sigma2_minmax = (sigma2 - sigma2.min()) / (sigma2.max() - sigma2.min())
                    logger.write_images({'sigma2_est_minmax': sigma2_minmax * 255})
                    sigma = torch.sqrt(sigma2)
                    logger.write_images({'sigma_est': sigma * 255})
                    sigma_minmax = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                    logger.write_images({'sigma_est_minmax': sigma_minmax * 255})
                model = weight_freezer(model, args)

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = f'{args.checkpoints}/{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--occlusion_module', type=str, default=None,
                        choices=[None, 'separate', 'with_uncertainty', 'separate_with_uncertainty',
                                 'separate_with_uncertainty_upsample8',
                                 'separate_with_uncertainty_morelayers',
                                 'separate_with_uncertainty_upsample8_morelayers'])
    parser.add_argument('--freeze_optical_flow_training', action='store_true', help='freezes training of optical flow estimation module')
    parser.add_argument('--freeze_features_training', action='store_true', help='freezes training of image features')
    parser.add_argument('--uncertainty_loss', type=str, default='huber',
                        choices=['huber', 'L2', 'huber_epe_direct',
                                 'huber_epe_direct_non_occluded', 'huber_non_occluded', 'L2_non_occluded'])
    parser.add_argument('--optical_flow_loss', type=str, default='L1',
                        choices=['L1', 'L1_non_occluded', 'L1_occluded_to_epe3'])

    parser.add_argument('--weighting_unc_loss', action='store_true', help='reweigting unc loss according epe sintel distribution')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dashcam_augmenentation', action='store_true')
    parser.add_argument('--blend_source', default='/datagrid/public_datasets/COCO/train2017', help="path to blending images")
    parser.add_argument('--normalized_features', help='normalize features before costvolume', action='store_true')
    parser.add_argument('--seed', help='', type=int, default=1234)
    parser.add_argument('--checkpoints', help='checkpoint directory', default='checkpoints')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = f'@{sys.argv[1]}'
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.isdir(args.checkpoints):
        os.mkdir(args.checkpoints)

    train(args)
