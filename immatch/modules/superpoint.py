import torch
import numpy as np

from third_party.superglue.models.superpoint import SuperPoint as SP
from third_party.superglue.models.utils import read_image
from .base import FeatureDetection, Matching

from immatch.utils.data_io import resize_im,load_gray_scale_tensor
import cv2
from PIL import Image

class SuperPoint(FeatureDetection, Matching):
    def __init__(self, args=None):   
        super().__init__()
        print(args)
        self.imsize = args['imsize']
        self.match_threshold = args['match_threshold'] if 'match_threshold' in args else 0.0
        self.model = SP(args).eval().to(self.device)
        self.dfactor=args['dfactor']
        rad = self.model.config['nms_radius']        
        self.name = f'SuperPoint_r{rad}'
        print(f'Initialize {self.name}')
        
    def load_and_extract(self, im_path):
        gray, scale=load_gray_scale_tensor(im_path,self.device,self.imsize,self.dfactor)
        #_, gray, scale = read_image(im_path, self.device, [self.imsize], 0, True)
        kpts, desc = self.extract_features(gray) 
        kpts = kpts * torch.tensor(scale).to(kpts) # N, 2
        return kpts, desc        
    
    def extract_features(self, gray):
        # SuperPoint outputs: {keypoints, scores, descriptors}
        pred = self.model({'image': gray})
        kpts = pred['keypoints'][0]
        desc = pred['descriptors'][0].permute(1, 0)  # N, D
        return kpts, desc
    
    def detect(self, gray):
        kpts, _ = self.extract_features(gray)
        return kpts
    
    def match_inputs_(self, gray1, gray2):
        kpts1, desc1 = self.extract_features(gray1)
        kpts2, desc2 = self.extract_features(gray2)
        kpts1 = kpts1.cpu().data.numpy()
        kpts2 = kpts2.cpu().data.numpy()
        
        # NN Match
        match_ids, scores = self.mutual_nn_match(desc1, desc2, threshold=self.match_threshold)
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores
    
    def match_pairs(self, im1_path, im2_path):
        gray1, sc1=load_gray_scale_tensor(im1_path,self.device,self.imsize,dfactor=16,value_to_scale=max)
        gray2, sc2=load_gray_scale_tensor(im2_path,self.device,self.imsize,dfactor=16,value_to_scale=max)   
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
    
    def match(self, gray1, gray2):
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        return matches, kpts1, kpts2, scores
        

