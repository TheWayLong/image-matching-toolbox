from argparse import Namespace
import torch
import numpy as np
import cv2
import torch.nn.functional as F

from PIL import Image
from immatch.utils.data_io import resize_im
from third_party.DFM.DeepFeatureMatcher import DeepFeatureMatcher
from .base import Matching

class DFM(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.args = args      
        # Load model

        self.model =DeepFeatureMatcher(enable_two_stage=args.two_stage, model=args.model, fine_model=args.fine_model,
                            ratio_th=args.ratio_th, fine_ratio=args.fine_ratio, bidirectional=args.bidirectional,
                            device=self.device, up_sample=args.up_sample, fine_weights_path=args.fine_weights_path)
        self.model = self.model.eval().to(self.device)
        self.name ='DFM_R2D2_TH'

    def load_im(self, im_path):
        im = Image.open(im_path).convert('RGB')
        w1, h1 = im.size
        w1,h1,scale1=resize_im(w1, h1, imsize=self.imsize, dfactor=16, value_to_scale=max, enforce=True)
        im=cv2.resize(np.array(im),(int(w1),int(h1)))

        return np.array(im)
    
    def match_inputs_(self, RGB1, RGB2):
        matches=self.model.match(RGB1, RGB2).detach().cpu().numpy()
        kpts1,kpts2=matches[:,:2],matches[:,2:4]
        return matches, kpts1, kpts2, None

    def match_pairs(self, im1_path, im2_path):
        try:
            RGB1= self.load_im(im1_path)
            RGB2 = self.load_im(im2_path)
        except:
            RGB1=im1_path
            RGB2=im2_path
        #print(RGB1.shape, RGB2.shape)
        matches, kpts1, kpts2, scores = self.match_inputs_(RGB1, RGB2)
        #cv2.imwrite(r'/remote-home/zwlong/image-matching-toolbox/test_match2.jpg',cv2.resize(draw_matches(np.array(gray1) , np.array(gray2) , kpts1, kpts2),(1600,1200)))
        return matches, kpts1, kpts2, scores

