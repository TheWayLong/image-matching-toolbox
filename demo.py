import immatch.matcher as mr

loftr=mr.LoFTR()
SG=mr.SuperGlue('Indoor')
SP=mr.SuperPoint()
SIFT=mr.SIFT_nn()
R2D2=mr.R2D2_nn()

img_path1=.../x.jpg
img_path2=.../.../x.jpg

matches,kpts1,kpts2,scores=loftr.match_pairs(img_path1,img_path2)
