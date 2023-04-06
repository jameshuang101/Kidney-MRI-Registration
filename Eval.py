import os

# For titan 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import tensorflow as tf
import scipy
import h5py
import numpy as np
import nibabel as nib
import pickle
import matplotlib.pyplot as plt
import cv2
import random
import config
import voxelmorph as vxm
import neurite as ne
import ants
import SimpleITK as sitk
import itk
from skimage import data, transform, exposure, img_as_float
from skimage.util import compare_images
from PIL import Image
from Train2D_Affine import grabFrameSegImage
from Display import auto_canny

#image1 is fixed image previous frame
#image2 is moving/registered image next frame
def volume_dice(vol, peak_frame):
    sum = 0
    for i in range(vol.shape[0]):
        sum += dice(vol[i], vol[peak_frame])

    sum -= 1
    return (sum / (vol.shape[0]-1))

def consec_dice(vol):
    sum = 0
    for i in range(1,vol.shape[0]):
        sum += dice(vol[i], vol[i-1])
    return (sum / (vol.shape[0]-1))

def dice(image1, image2):
    # shape of image2 and image1_bin: (height, width)
    intersection = np.sum(image1 * image2)
    if (np.sum(image2)==0) and (np.sum(image1)==0):
        return 1
    return (2*intersection) / (np.sum(image2) + np.sum(image1))


def volume_hausdorff(vol, peak_frame):
    sum = 0
    for i in range(vol.shape[0]):
        sum += hausdorff(vol[i], vol[peak_frame])

    return (sum / (vol.shape[0]-1))

def consec_hausdorff(vol):
    sum = 0
    for i in range(1,vol.shape[0]):
        sum += hausdorff(vol[i], vol[i-1])

    return (sum / (vol.shape[0]-1))

def hausdorff(image1, image2):
    from scipy.spatial.distance import directed_hausdorff
    return directed_hausdorff(image1, image2)[0]

def volume_mae(vol, peak_frame):
    sum = 0
    for i in range(vol.shape[0]):
        sum += mae(vol[i], vol[peak_frame])
    return (sum / (vol.shape[0]-1))

def consec_mae(vol):
    sum = 0
    for i in range(1, vol.shape[0]):
        sum += mae(vol[i], vol[i-1])
    return (sum / (vol.shape[0]-1))

def mae(image1, image2):
    image1 = np.array(image1).ravel()
    image2 = np.array(image2).ravel()
    n = image1.shape[0]
    sum = 0
    for i in range(n):
        sum += abs(image1[i] - image2[i])
    return (sum/n)

def alpha_blend(image1, image2, alpha=0.5, mask1=None,  mask2=None):
    return ((1 - alpha) * image1 + alpha * image2)

def checkerboard(image1, image2, n_squares = 8):
    im = compare_images(image1, image2, method='checkerboard', n_tiles=(n_squares,n_squares))
    return img_as_float(im)

def imsubtract(image1, image2):
    return cv2.subtract(image1, image2)

#Probably will not need, just use boundary overlay
def seg_overlay(image1, seg2r=None):
    seg2r = grabFrameSegImage(4, 14, 20)
    seg2r *= 255
    seg2r = cv2.resize(seg2r, (384, 384))
    seg2r = rgb2gray(seg2r)
    return alpha_blend(image1, seg2r, alpha=0.3)

#seg2r is segmentation of registered
#seg2f is segmentation of moving
def boundary_overlay(image1, seg2r=None, seg2m=None):
    #get boundaries of seg maps
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    seg2r_border = auto_canny(seg2r)
    seg2r_r = cv2.merge([0*seg2r_border, seg2r_border, 0*seg2r_border])
    #seg2m_border = auto_canny(seg2m)
    #seg2m_b = cv2.merge([seg2m_border, 0*seg2m_border, seg2m_border])
    seg2m_b = cv2.merge([seg2m, 0*seg2m, seg2m])
    return cv2.add(image1_rgb, cv2.add(seg2r_r, seg2m_b))

def save_volume(vol1, vol2, mode='orig', name=config.PATIENT_NAME, peak_frame=config.PEAK_FRAME):
    shape = vol1.shape
    vol1_255 = (vol1*255).astype('uint8')
    vol2_255 = (vol2*255).astype('uint8')
    alpha_vol = np.zeros((shape)).astype('float')
    checker_vol = np.zeros((shape)).astype('float')
    sub_vol = np.zeros((shape)).astype('float')
    for i in range(shape[0]):
        alpha_vol[i] = alpha_blend(vol1[peak_frame],vol2[i])
        checker_vol[i] = checkerboard(vol1_255[peak_frame],vol2_255[i])
        sub_vol[i] = imsubtract(vol1_255[peak_frame],vol2_255[i])
    #checker_vol /= 255.
    sub_vol /= 255.
    np.save(os.path.join(config.EVAL_PATH, config.PATIENT_NAME + str(mode) + '_alpha_pk.npy'), alpha_vol)
    np.save(os.path.join(config.EVAL_PATH, config.PATIENT_NAME + str(mode) + '_checker_pk.npy'), checker_vol)
    np.save(os.path.join(config.EVAL_PATH, config.PATIENT_NAME + str(mode) + '_subtract_pk.npy'), sub_vol)

def save_bound(vol1, seg1, seg2, mode='orig', name = config.PATIENT_NAME, peak_frame=config.PEAK_FRAME):
    shape = vol1.shape
    vol1_255 = (vol1*255).astype('uint8')
    seg1_255 = (seg1*255).astype('uint8')
    seg2_255 = (seg2*255).astype('uint8')
    bound_vol = np.zeros([*vol1.shape, 3]).astype('float')
    for i in range(shape[0]):
        bound_vol[i] = boundary_overlay(vol1_255[i],seg1_255[i], seg2_255[i])
    bound_vol /= 255.
    np.save(os.path.join(config.EVAL_PATH, config.PATIENT_NAME + '_bound.npy'), bound_vol)

def main():
    seg1 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_moving_segs.npy'))
    print('Orig dice: ' + str(volume_dice(seg1, config.PEAK_FRAME)))
    seg2 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_segs.npy'))
    print('Affine dice: ' + str(volume_dice(seg2, config.PEAK_FRAME)))
    seg3 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_deformed_segs.npy'))
    print('Deform dice: ' + str(volume_dice(seg3, config.PEAK_FRAME)))

    print('Orig haus: ' + str(volume_hausdorff(seg1, config.PEAK_FRAME)))
    print('Affine haus: ' + str(volume_hausdorff(seg2, config.PEAK_FRAME)))
    print('Deform haus: ' + str(volume_hausdorff(seg3, config.PEAK_FRAME)))

    volumef = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_fixed_images.npy'))
    volume1 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_moving_images.npy'))
    print('Orig mae: ' + str(volume_mae(volume1, config.PEAK_FRAME)))
    volume2 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_images.npy'))
    print('Affine mae: ' + str(volume_mae(volume2, config.PEAK_FRAME)))
    volume3 = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_deformed_images.npy'))
    print('Deform mae: ' + str(volume_mae(volume3, config.PEAK_FRAME)))

    print('Consec Orig dice: ' + str(consec_dice(seg1)))
    print('Consec Aff dice: ' + str(consec_dice(seg2)))
    print('Consec Deform dice: ' + str(consec_dice(seg3)))

    print('Consec orig Haus: ' + str(consec_hausdorff(seg1)))
    print('Consec af Haus: ' + str(consec_hausdorff(seg2)))
    print('Consec def Haus: ' + str(consec_hausdorff(seg3)))

    print('Consec orig mae: ' + str(consec_mae(volume1)))
    print('Consec af mae: ' + str(consec_mae(volume2)))
    print('Consec defs mae: ' + str(consec_mae(volume3)))
    # save_volume(volumef, volume1)
    # save_volume(volumef, volume3, mode='reg')
    #save_bound(volume3, seg1, seg3)


if __name__ == '__main__':
    #checkerboard()
    main()
