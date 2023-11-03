from immatch.utils.model_helper import init_matcher

def LoFTR(benchmark_name='Outdoor'):
    #options:
    #Outdoor|Indoor#
    model, model_conf=init_matcher('loftr',benchmark_name)
    print(model_conf)
    return model
def R2D2_nn():
    #options:
    #Outdoor#
    model, model_conf=init_matcher('r2d2','default')
    print(model_conf)
    return model
def SIFT_nn():
    #options:
    #all scenes#
    model, model_conf=init_matcher('sift','default')
    print(model_conf)
    return model
def SuperPoint():
    #options:
    #Outdoor#
    model, model_conf=init_matcher('superpoint','default')
    print(model_conf)
    return model

def SuperGlue(benchmark_name='Outdoor'):
    #options:
    #Outdoor|Indoor#
    model, model_conf=init_matcher('superglue',benchmark_name)
    print(model_conf)
    return model

LoFTR()
SuperGlue('Indoor')
SuperPoint()
SIFT_nn()
R2D2_nn()