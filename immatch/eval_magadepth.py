from collections import defaultdict
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import argparse
import os
import cv2
from PIL import Image
import matplotlib.cm as cm
from immatch.utils.model_helper import init_model
from immatch.utils.sg_utils import make_matching_plot
import immatch.utils.metrics as M
from immatch.utils.data_io import resize_im

def draw_matches(img_A, img_B, keypoints0, keypoints1):
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
        p1s.append(cv2.KeyPoint(int(x1), int(y1), 1))
        p2s.append(cv2.KeyPoint(int(keypoints1[i][0]), int(keypoints1[i][1]), 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))

    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s,
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
def draw_matches_precision(img_A, img_B, keypoints0, keypoints1,color):
   

    return matched_images
def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, R,T, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ R

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def compute_relapose_aspan(kpts0, kpts1, K0, K1, pix_thres=0.5, conf=0.99999):
    """ Original code from ASpanFormer repo:
        https://github.com/apple/ml-aspanformer/blob/main/src/utils/metrics.py
    """

    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = pix_thres / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def load_megadepth_pairs_npz(npz_root, npz_list):
    with open(npz_list, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    print(f"Parse {len(npz_names)} npz from {npz_list}.")

    pairs = []
    for name in npz_names:
        scene_info = np.load(f"{npz_root}/{name}.npz", allow_pickle=True)

        # Collect pairs
        for pair_info in scene_info['pair_infos']:
            (id1, id2), overlap, _ = pair_info
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)

            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
            pairs.append(Namespace(
                im1=im1, im2=im2, overlap=overlap, 
                K1=K1, K2=K2, t=T12[:3, 3], R=T12[:3, :3]
            ))
    print(f"Loaded {len(pairs)} pairs.")
    return pairs

def eval_megadepth_relapose(
    matcher,
    data_root,
    npz_root,
    npz_list,
    method='',    
    ransac_thres=0.5,
    thresholds=[1, 3, 5, 10, 20],
    print_out=False,
    debug=False,
):
    statis = defaultdict(list)
    np.set_printoptions(precision=2)
    im1_list=[]
    # Load pairs
    pairs = load_megadepth_pairs_npz(npz_root, npz_list)    
    # Eval on pairs
    print(f">>> Start eval on Megadepth: method={method} rthres={ransac_thres} ... \n")
    for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
        if debug and i > 10:
            break
        
        '''
        if (i==1246 or 
           i ==61 or 
           i==90 or 
           i== 147 or
           i==282 or
           i==296 or 
           i==340 or 
           i== 494 or
           i==703 or
           i==741 or
           i==752 or
           i==848)is not True :
            continue
        '''
        K1 = pair.K1
        K2 = pair.K2
        t_gt = pair.t
        R_gt = pair.R
        im1 = str(data_root +r'/' +pair.im1)
        im2 = str(data_root +r'/'+ pair.im2)
        #if( im1!=r"/remote-home/zwlong/image_matching_benchmark/data/megadepth/Undistorted_SfM/0022/images/2142521640_9a08bee026_o.jpg"):
        #    continue

        img1 = Image.open(im1)
        w1, h1 = img1.size
        img2 = Image.open(im2)
        w2, h2 = img2.size
        
        #w1,h1,scale1=resize_im(w1, h1, imsize=1200, dfactor=16, value_to_scale=max, enforce=True)
        #w2,h2,scale2=resize_im(w2, h2, imsize=1200, dfactor=16, value_to_scale=max, enforce=True)
        matches, pts1, pts2, scores = matcher(im1, im2)
        pts1,pts2=matches[:, :2], matches[:, 2:4]
        
        #K1[0] = K1[0] / scale1[0]
       # K1[1] = K1[1] / scale1[1]
        #K2[0] = K2[0] / scale2[0]
        #K2[1] = K2[1] / scale2[1]
        #w1,h1,w2,h2=w1/scale1[0],h1/ scale1[1],w2/ scale2[0],h2/ scale2[1]
        img1_test=np.array(img1)
        #img1_test=cv2.resize(img1_test,(int(w1),int(h1)))
        img2_test=np.array(img2)
        #img2_test=cv2.resize(img2_test,(int(w2),int(h2)))
        
        
        #print(h1,w1,h2,w2)
        epi_errs = compute_epipolar_error(pts1, pts2, R_gt,t_gt, K1, K2)
        correct = epi_errs < 1e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        
        
        color = cm.jet(correct)
        color[correct] = [0, 1, 0, 1]
        color[(1*correct)<0.5] = [1, 0, 0, 1]
        if i%50==0:
            make_matching_plot(image0=np.array(img1_test), 
                           image1 =np.array(img2_test), 
                           kpts0=None, 
                           kpts1=None, 
                           mkpts0=matches[:, :2], 
                           mkpts1=matches[:, 2:4],
                           color=color,
                           path=r'/remote-home/zwlong/image-matching-toolbox/outputs/viz/mega_loftr/'+str(i)+'.jpg',
                           text=[r'inliers/correspondence:'+str(num_correct)+ r'/' + str(matches.shape[0]) ,]
                           )
                        
        #cv2.imwrite(r'./test_match_dfm.jpg',draw_matches_precision(np.array(img1_test), np.array(img2_test), matches[:, :2], matches[:, 2:4],color))
        
        
        statis['precisions'].append(precision)
        # Compute pose errors
        ret = compute_relapose_aspan(
            pts1, pts2, K1, K2, pix_thres=ransac_thres
        )
        
        if ret is None:
            statis['failed'].append(i)
            statis['R_errs'].append(np.inf)
            statis['t_errs'].append(np.inf)
            statis['inliers'].append(np.array([]).astype(bool ))
        else:
            R, t, inliers = ret
            R_err, t_err = M.cal_relapose_error(R, R_gt, t, t_gt)
            statis['R_errs'].append(R_err)
            statis['t_errs'].append(t_err)
            statis['inliers'].append(inliers.sum() / len(pts1))
            if print_out:
                print(f"#M={len(matches)} R={R_err:.3f}, t={t_err:.3f}")
    
    precisions=np.array(statis['precisions'])
    precisions=np.mean(precisions)
    print(f"Total samples: {len(pairs)} Failed:{len(statis['failed'])}.")
    pose_auc,pose_map = M.cal_relapose_auc(statis, thresholds=thresholds)
    print('Total precisions: ',precisions)
    return pose_auc,pose_map


def eval_megadepth(
    root_dir,
    method,
    benchmark='megadepth',
    ransac_thres=0.5,
    print_out=False,
    debug=False,
):
    
    # Init paths
    npz_root =  r'/remote-home/zwlong/image_matching_benchmark/data/megadepth/megadepth_test_1500_scene_info'
    npz_list =  r'/remote-home/zwlong/image_matching_benchmark/data/megadepth/megadepth_test_1500_scene_info/megadepth_test_1500.txt'
    data_root = r'/remote-home/zwlong/image_matching_benchmark/data/megadepth/Undistorted_SfM'
        
    # Init model
    model, config = init_model(method, benchmark, root_dir=root_dir)    
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Eval
    eval_megadepth_relapose(
        matcher, 
        data_root,
        npz_root,
        npz_list,
        model.name,
        print_out=print_out,
        debug=debug,
    )

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Localize Inloc')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--config', type=str, default=None)    
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--data_dir',type=str,default=r'/remote-home/zwlong/image_matching_benchmark/data/megadepth')
    parser.add_argument('--benchmark_name', type=str, default='megadepth')
    parser.add_argument('--root_dir', type=str, default='.')  
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    eval_megadepth(
        root_dir=args.root_dir,
        method=args.config,
        benchmark=args.benchmark_name,
        
    )
    
