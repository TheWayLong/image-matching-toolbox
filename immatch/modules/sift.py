from argparse import Namespace
import torch
import numpy as np
import cv2

from .base import FeatureDetection, Matching
from ..utils.data_io import read_im_gray

class SIFT(FeatureDetection, Matching):
    def __init__(self, args=None):   
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)        
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.model = cv2.SIFT_create(args.npts)        
        self.name = f'SIFT{args.npts}'
        self.dfactor=args.dfactor
        print(f'Initialize {self.name}')
        
    def load_im(self, im_path):
        im, scale = read_im_gray(im_path, self.imsize,dfactor=self.dfactor,value_to_scale=max)
        im = np.array(im)
        return im, scale

    def load_and_extract(self, im_path):
        im, scale = self.load_im(im_path)
        kpts, desc = self.extract_features(im)
        kpts = kpts * scale
        return kpts, desc        
    
    def extract_features(self, im):
        try:
            kpts, desc = self.model.detectAndCompute(im, None)
        except:
            im = cv2.normalize(im,None,0,255,cv2.NORM_MINMAX).astype('uint8')
            kpts, desc = self.model.detectAndCompute(im, None)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        return kpts, desc
    
    def load_and_detect(self, im_path):
        im, scale = self.load_im(im_path)
        kpts, desc = self.detect(im)
        kpts = kpts * scale        
        return kpts

    def detect(self, im):
        kpts = self.model.detect(im)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])        
        return kpts
    
    def match_inputs_(self, im1, im2):
        kpts1, desc1 = self.extract_features(im1)
        kpts2, desc2 = self.extract_features(im2)
        
        # NN Match
        match_ids, scores = self.mutual_nn_match(desc1, desc2, threshold=self.match_threshold)
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores
    
    def match_pairs(self, im1_path, im2_path):
        im1, sc1 = self.load_im(im1_path)
        im2, sc2 = self.load_im(im2_path)
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, im2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
    
    def match(self,im_gray1,im_gray2):
        matches, kpts1, kpts2, scores = self.match_inputs_(im_gray1, im_gray2)
        return matches, kpts1, kpts2, scores
