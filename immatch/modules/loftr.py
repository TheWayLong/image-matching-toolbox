from argparse import Namespace
import torch
import numpy as np
import cv2

from third_party.loftr.src.loftr import LoFTR as LoFTR_, default_cfg
from .base import Matching
from immatch.utils.data_io import load_gray_scale_tensor_cv, img2tensor,resize_im


class LoFTR(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        print('imsize=',self.imsize)    
        print(args)    
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale
        self.eval_coarse = args.eval_coarse
        self.dfactor= args.dfactor
        
        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        if args.match_type =='sinkhorn':
            conf['match_coarse']['match_type']='sinkhorn'
            conf['match_coarse']['sparse_spvs']=False
    
        self.model = LoFTR_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'LoFTR_{self.ckpt_name}'        
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')
        
    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor= self.dfactor,value_to_scale=max
        )

    def match_inputs_(self, gray_tensor1, gray_tensor2):
        batch = {'image0': gray1, 'image1': gray2}
        self.model(batch)
        if self.eval_coarse:
            kpts1 = batch['mkpts0_c'].cpu().numpy()
            kpts2 = batch['mkpts1_c'].cpu().numpy()
        else:
            kpts1 = batch['mkpts0_f'].cpu().numpy()
            kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        gray1, sc1, _ = self.load_im(im1_path)
        gray2, sc2, _ = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
    
    def match(self,gray1,gray2):
        h1,w1,h2,w2=gray1.shape,gray2.shape
        h1,w1,scale1=resize_im(h1,w1,-1,dfactor=16)
        h2,w2,scale2=resize_im(h2,w2,-1,dfactor=16)
        gray1 = gray1.resize((w1, h1), Image.BICUBIC)
        gray2 = gray1.resize((w2, h2), Image.BICUBIC)
        gray1= img2tensor(gray1)
        gray2= img2tensor(gray2)
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        upscale = np.array([scale1 + scale2])
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
    
    
    
    
        
    
    
    
