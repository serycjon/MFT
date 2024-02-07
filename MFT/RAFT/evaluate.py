import sys
sys.path.append('core')
import einops
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from MFT.RAFT.core import datasets
from MFT.RAFT.core.utils import flow_viz
from MFT.RAFT.core.utils import frame_utils

from MFT.RAFT.core.raft import RAFT
from MFT.RAFT.core.utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        outputs = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = outputs['flow']
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

def uncertainty_loss(uncertainty_est, flow_est, flow_gt):
    huber_loss = torch.nn.SmoothL1Loss(reduction='none')
    alpha = uncertainty_est
    exp_alpha = torch.exp(-alpha)
    loss = exp_alpha * huber_loss(flow_est, flow_gt) + 0.5 * alpha
    return loss.cpu()

def occlusion_loss(occl_est, occl_gt):
    occl_gt_thresholded = occl_gt > 0.5
    occl_gt_thresholded = occl_gt_thresholded[0,:,:].long()
    cross_ent_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss = cross_ent_loss(torch.unsqueeze(occl_est.softmax(dim=0), dim=0), torch.unsqueeze(occl_gt_thresholded, dim=0))
    return torch.squeeze(loss).cpu()

def occlusion_accuracy(occl_est, occl_gt):
    pred = occl_est.softmax(dim=1)[1, :, :] > 0.5
    gt = occl_gt[0, :, :]
    accuracy = float((pred == gt).float().mean())
    return accuracy

def uncertainty_eval(uncertainty_est, flow_est, flow_gt):
    gt_epe = einops.reduce(torch.square(flow_est - flow_gt),
                           'xy H W -> 1 H W',
                           reduction='sum')
    pred_epe = uncertainty_est

    overshoot = (pred_epe > gt_epe).float().mean()
    diff = torch.abs(gt_epe - pred_epe)
    sub_1 = (diff < 1).float().mean()
    sub_5 = (diff < 5).float().mean()
    return overshoot, sub_1, sub_5

@torch.no_grad()
def validate_sintel(model, iters=12, n_val=None, subsplit=None, quiet=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, subsplit=subsplit, load_occlusion=True)
        epe_list = []
        uncer_loss_list = []
        occl_loss_list = []
        occl_accuracy_list = []
        uncer_overshoot_list = []
        uncer_sub_1px_list = []
        uncer_sub_5px_list = []
        

        for val_id in range(len(val_dataset)):
            if (n_val is not None) and (val_id >= n_val):
                break
            image1, image2, flow_gt, _, occl_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            prediction_dict = model(image1, image2, iters=iters, test_mode=True)
            flow_low, flow_pr = prediction_dict['coords'], prediction_dict['flow']

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if model.uncertainty_estimation:
                uncertainty_pr = prediction_dict['uncertainty']
                uncertainty = padder.unpad(uncertainty_pr[0]).cpu()
                uncer_loss = uncertainty_loss(uncertainty, flow, flow_gt)
                uncer_loss_list.append(uncer_loss.view(-1).numpy())

                overshoot, sub_1, sub_5 = uncertainty_eval(uncertainty, flow, flow_gt)
                uncer_overshoot_list.append(overshoot)
                uncer_sub_1px_list.append(sub_1)
                uncer_sub_5px_list.append(sub_5)

            if model.occlusion_estimation:
                occl_pr = prediction_dict['occlusion']
                occlusion = padder.unpad(occl_pr[0]).cpu()
                occl_loss = occlusion_loss(occlusion, occl_gt)
                occl_loss_list.append(occl_loss.view(-1).numpy())

                occl_accuracy_list.append(occlusion_accuracy(occlusion, occl_gt))

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        if not quiet:
            print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f'eval/flow {dstype}'] = np.mean(epe_list)

        if model.uncertainty_estimation:
            # uncer_all = np.concatenate(uncer_loss_list)
            # uncert_mean = np.mean(uncer_all)
            # results[f'eval/uncertainty loss {dstype}'] = uncert_mean

            overshoot = np.mean(uncer_overshoot_list)
            sub_1 = np.mean(uncer_sub_1px_list)
            sub_5 = np.mean(uncer_sub_5px_list)
            results[f'eval/uncertainty overshoot {dstype}'] = overshoot
            results[f'eval/uncertainty sub_1 {dstype}'] = sub_1
            results[f'eval/uncertainty sub_5 {dstype}'] = sub_5
        if model.occlusion_estimation:
            occl_all = np.concatenate(occl_loss_list)
            occl_mean = np.mean(occl_all)
            # print("Validation (%s) OCCL: %f, Uncertainty %f" % (dstype, occl_mean, uncert_mean))
            results[f'eval/occl loss {dstype}'] = occl_mean

            occl_mean = np.mean(occl_accuracy_list)
            if not quiet:
                print("Validation (%s) OCCL_acc: %f, EPE overshoot: %f, sub1: %f, sub5: %f" % (dstype, occl_mean,
                                                                                               overshoot, sub_1, sub_5))
            results[f'eval/occl acc {dstype}'] = occl_mean

    return results

@torch.no_grad()
def validate_kubric(model, iters=12, n_val=20, subsplit=None, quiet=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    val_dataset = datasets.KubricDataset(split=subsplit, load_occlusion=True, correct_flow=True)

    deltas = [1, 2, 4, 8, 16]
    for delta in deltas:
        epe_list = []
        uncer_loss_list = []
        occl_loss_list = []
        occl_accuracy_list = []
        uncer_overshoot_list = []
        uncer_sub_1px_list = []
        uncer_sub_5px_list = []

        for val_id in range(n_val):
            image1, image2, flow_gt, valid, occl_gt = val_dataset.get_data_delta(val_id, delta)
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            prediction_dict = model(image1, image2, iters=iters, test_mode=True)
            flow_low, flow_pr = prediction_dict['coords'], prediction_dict['flow']

            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if model.uncertainty_estimation:
                uncertainty_pr = prediction_dict['uncertainty']
                uncertainty = padder.unpad(uncertainty_pr[0]).cpu()
                uncer_loss = uncertainty_loss(uncertainty, flow, flow_gt)
                uncer_loss_list.append(uncer_loss.view(-1).numpy())

                overshoot, sub_1, sub_5 = uncertainty_eval(uncertainty, flow, flow_gt)
                uncer_overshoot_list.append(overshoot)
                uncer_sub_1px_list.append(sub_1)
                uncer_sub_5px_list.append(sub_5)

            if model.occlusion_estimation:
                occl_pr = prediction_dict['occlusion']
                occlusion = padder.unpad(occl_pr[0]).cpu()
                occl_loss = occlusion_loss(occlusion, occl_gt)
                occl_loss_list.append(occl_loss.view(-1).numpy())
                occl_accuracy_list.append(occlusion_accuracy(occlusion, occl_gt))

        # print(f"occl_accuracy_list: {occl_accuracy_list}")

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        if not quiet:
            print("Validation delta: (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (delta, epe, px1, px3, px5))
        results[f'eval/flow delta {delta}'] = np.mean(epe_list)

        if model.uncertainty_estimation:
            # uncer_all = np.concatenate(uncer_loss_list)
            # uncert_mean = np.mean(uncer_all)
            # results[f'eval/uncertainty loss delta {delta}'] = uncert_mean

            overshoot = np.mean(uncer_overshoot_list)
            sub_1 = np.mean(uncer_sub_1px_list)
            sub_5 = np.mean(uncer_sub_5px_list)
            results[f'eval/uncertainty overshoot delta {delta}'] = overshoot
            results[f'eval/uncertainty sub_1 delta {delta}'] = sub_1
            results[f'eval/uncertainty sub_5 delta {delta}'] = sub_5
        if model.occlusion_estimation:
            occl_all = np.concatenate(occl_loss_list)
            occl_mean = np.mean(occl_all)
            # print("Validation delta: (%s) OCCL: %f, Uncertainty %f" % (delta, occl_mean, uncert_mean))
            # results[f'eval/occl loss delta {delta}'] = occl_mean

            # occl_acc = np.concatenate(occl_accuracy_list)
            occl_mean = np.mean(occl_accuracy_list)
            results[f'eval/occl acc delta {delta}'] = occl_mean
            # print("Validation delta: (%s) OCCL-acc: %f, Uncertainty %f" % (delta, occl_mean, uncert_mean))
            if not quiet:
                print("Validation delta: (%s) OCCL_acc: %f, EPE overshoot: %f, sub1: %f, sub5: %f" % (delta, occl_mean,
                                                                                                      overshoot, sub_1, sub_5))

    return results


@torch.no_grad()
def validate_viper(model, iters=32):
    """ Peform validation using the VIPER (train) split - viper_split.txt """
    model.eval()
    val_dataset = datasets.VIPER(split='validation')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='viper')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.detach().cpu().numpy().reshape(-1)
        mag = mag.detach().cpu().numpy().reshape(-1)
        val = valid_gt.detach().cpu().numpy().reshape(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).astype(np.float32)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val])

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epe_all = np.concatenate(epe_list)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation VIPER: %f, %f, %f, %f, %f" % (epe, f1, px1, px3, px5))
    return {'viper-epe': epe, 'viper-f1': f1, 'viper-px1': px1, 'viper-px3': px3, 'viper-px5': px5}



@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'viper':
            validate_viper(model.module)


