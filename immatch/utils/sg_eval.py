#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import numpy as np
import glob
import time
import pydegensac
import cv2
from PIL import Image
from tqdm import tqdm
from immatch.utils.model_helper import init_model
from immatch.utils.data_io import resize_im
from immatch.utils.sg_utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def draw_matches(img_A, img_B, keypoints0, keypoints1):
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
        p1s.append(cv2.KeyPoint(int(x1), int(y1), 1))
        p2s.append(cv2.KeyPoint(int(keypoints1[i][0]), int(keypoints1[i][1]), 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
    try:
        matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s,
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    except:
        img_A = cv2.normalize(img_A,None,0,255,cv2.NORM_MINMAX).astype('uint8')
        img_B = cv2.normalize(img_B,None,0,255,cv2.NORM_MINMAX).astype('uint8')
        matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s,
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    return matched_images

def eval_scannet_relapose(
    matcher,
    input_pairs,
    input_dir,
    output_dir,
    method='',    
    
):
    
    print(f">>> Start eval on scannet: method={method} rthres={0.5} ... \n")
    
    with open(input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    random.Random(0).shuffle(pairs)


    if not all([len(p) == 38 for p in pairs]):
        raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format( input_pairs))


    matching=matcher
    
    input_dir = Path( input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path( output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    IM1_LIST=[]
    for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        
                # Load the image pair.
     # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
        else:
                rot0, rot1 = 0, 0
        # Perform the matching.        
        img_path0=input_dir / name0
        IM1_LIST.append(img_path0)
        
        img_path1=input_dir / name1
        #test
        image0, inp0, scales0 = read_image(
            input_dir / name0, torch.device('cuda:0'),  [640,480], rot0, 1)
        image1, inp1, scales1 = read_image(
            input_dir / name1, torch.device('cuda:0'), [640,480], rot1, 1)
        
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        #test
        matches,kpts0, kpts1,conf = matching(inp0, inp1)
        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
        np.savez(str(matches_path), **out_matches)
        
        # Keep the matching keypoints.

        mkpts0 = matches[:,:2]
        mkpts1 = matches[:,2:4]
        cv2.imwrite(r'/remote-home/zwlong/image_matching_benchmark/test.jpg',cv2.resize(draw_matches(np.array(image0), np.array(image1), mkpts0, mkpts1),(1600,1200)))
        
        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Write the evaluation results to disk.
        out_eval = {'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs}
        np.savez(str(eval_path), **out_eval)
        
        # Collate the results into a final table and print to terminal.
    
 
    
    pose_errors = []
    precisions = []
    matching_scores = []
    for pair in pairs:
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        eval_path = output_dir / \
            '{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
        
    

def eval_yfcc_relapose(
    matcher,
    input_pairs,
    input_dir,
    output_dir,
    method='',    
    
):
    
    print(f">>> Start eval on scannet: method={method} rthres={0.5} ... \n")
    
    with open(input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    #random.Random(0).shuffle(pairs)


    if not all([len(p) == 38 for p in pairs]):
        raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format( input_pairs))


    matching=matcher
    
    input_dir = Path( input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path( output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    
    for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        
                # Load the image pair.
     # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
        else:
                rot0, rot1 = 0, 0
        # Perform the matching.        
        img_path0=input_dir / name0
        img_path1=input_dir / name1
        #test
        image0, inp0, scales0 = read_image(
            input_dir / name0, torch.device('cuda:0'),  [1600], rot0, 1,color='gray',dfactor=16)
        image1, inp1, scales1 = read_image(
            input_dir / name1, torch.device('cuda:0'), [1600], rot1, 1,color='gray',dfactor=16)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        #test
        #matches,kpts0, kpts1,conf = matching(image0.copy(), image1.copy())
        matches,kpts0, kpts1,conf = matching(inp0, inp1)
        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
        np.savez(str(matches_path), **out_matches)
        
        # Keep the matching keypoints.

        mkpts0 = matches[:,:2]
        mkpts1 = matches[:,2:4]
        cv2.imwrite(r'/remote-home/zwlong/image_matching_benchmark/test.jpg',cv2.resize(draw_matches(np.array(image0), np.array(image1), mkpts0, mkpts1),(1600,1200)))
        
        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Write the evaluation results to disk.
        out_eval = {'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs}
        np.savez(str(eval_path), **out_eval)
        
        # Collate the results into a final table and print to terminal.
    pose_errors = []
    precisions = []
    matching_scores = []
    for pair in pairs:
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        eval_path = output_dir / \
            '{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
        
    




