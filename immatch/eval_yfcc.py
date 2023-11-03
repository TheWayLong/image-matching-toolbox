from collections import defaultdict
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import argparse
import os
import cv2
import os.path as osp
from PIL import Image

from immatch.utils.model_helper import init_model
import immatch.utils.metrics as M
from immatch.utils.data_io import resize_im
from immatch.utils.sg_eval import eval_yfcc_relapose

def eval_yfcc(
    root_dir,
    method,
    benchmark,
    input_pairs,
    input_dir,
    output_dir,
    
):
    
    # Init model
    model, config = init_model(method, benchmark, root_dir=root_dir)    
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Eval
    eval_yfcc_relapose(
        matcher, 
        input_pairs,
        input_dir,
        output_dir,
        model.name,
    )

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Localize Inloc')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--config', type=str, default=None)    
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument(
        '--input_pairs', type=str, default=r'/remote-home/zwlong/image-matching-toolbox/data/yfcc_test_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default=r'/remote-home/zwlong/image-matching-toolbox/data/datasets/raw_data/yfcc100m/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default=r'/remote-home/zwlong/image-matching-toolbox/outputs/yfcc/dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument('--benchmark_name', type=str, default='yfcc')
    parser.add_argument('--root_dir', type=str, default='.')  
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    eval_yfcc(
        root_dir=args.root_dir,
        method=args.config,
        benchmark=args.benchmark_name,
        input_pairs=args.input_pairs,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    
