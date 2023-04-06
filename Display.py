import os

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
import nibabel as nb
import scipy.ndimage as nd
from skimage import morphology
from PIL import Image

EDGES = True

def show_edges(image, minVal=70, maxVal=120):
    image_e = cv2.GaussianBlur(image, (3,3), 0)
    #image_e = cv2.Canny(image,minVal,maxVal)
    #image_e = cv2.addWeighted(image, 1.5, image_g, -0.5, 0, image)
    #return cv2.add(image_e, image)[...,np.newaxis]
    #image_e = morphology.remove_small_objects(image_e, 10)
    return image_e

def auto_canny(image, sigma=0.33):
    v = np.average(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    print(lower)
    print(upper)
    return edged

def edge_enhance(image):
    image_g = nd.gaussian_laplace(image, sigma=3, output= np.dtype('float32'))
    image = image.astype(np.dtype('float32'))
    enhanced = cv2.addWeighted(image, 1.5, image_g, -0.5, 0, image)
    return image_g

def adjust_gamma(image, gamma=1.5):
    image_g = image.astype('float32')/255.
    invGamma = 1.0/gamma
    image_g = image_g**invGamma
    return (image_g*255).astype('uint8')

def save_npy_vid(array, filename):
    array = np.squeeze(array)
    #print(array.shape)
    out = cv2.VideoWriter(os.path.join(config.OUTPUTS_PATH, filename.replace('.npy','.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 10,
            (config.IMG_WIDTH,config.IMG_HEIGHT), False)

    for i in range(len(array)):
        image = array[i]
        out.write(image)
    out.release()

def save_npy_gif(array, filename):
    array = np.squeeze(array)
    imgs = [Image.fromarray(img) for img in array]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(filename.replace('.npy','.gif'), save_all=True, append_images=imgs[1:], duration=100, loop=0)

def save_npy_frame(array, filename):
    frame = array
    print(frame.shape)
    im = Image.fromarray(frame)
    im.save(os.path.join(config.EFRAMES_PATH, filename + '.png'))

def save_all_frames():
    dir = os.listdir(config.EVAL_PATH)
    arrays = [x for x in dir if '.npy' in x]
    for name in arrays:
        vol = np.load(os.path.join(config.EVAL_PATH, name))
        vol = (vol*255).astype('uint8')
        for i in range(vol.shape[0]):
            new_name = name.replace('.npy','')
            frame = i+1
            new_name = new_name + '_Frame_' + str(frame).zfill(3)
            save_npy_frame(vol[i], new_name)

def save_all_vids():
    dir = os.listdir(config.EVAL_PATH)
    arrays = [x for x in dir if 'pk.npy' in x]
    for name in arrays:
        vol = np.load(os.path.join(config.EVAL_PATH, name))
        vol = (vol*255).astype('uint8')
        save_npy_gif(vol, os.path.join(config.GIF_PATH, name))
        #save_npy_vid(vol, os.path.join(config.VIDS_PATH, name))

def save_all_nii():
    dir = os.listdir(config.OUTPUTS_PATH)
    arrays = [x for x in dir if '.npy' in x]
    for name in arrays:
        vol = np.load(os.path.join(config.OUTPUTS_PATH, name))
        vol = (vol*255).astype('uint8')
        save_nii(vol, os.path.join(config.NII_PATH, name))

def save_nii(array, filename):
    array = np.squeeze(array)
    ni_img = nib.Nifti1Image(array, affine=np.eye(4))
    nib.save(ni_img, filename.replace('npy','nii'))

def main():
    save_all_frames()
    # seg_path = os.path.join(config.HOME_PATH, 'segmentations2d_cropped')
    # segs = os.listdir(seg_path)
    # patient_nb = 3
    # slice_nb = 13
    # tag = 'Patient_' + str(patient_nb).zfill(3) + '_Slice_' + str(slice_nb).zfill(3)
    # seg_name = 'Seg_' + tag + '.npy'
    # img_name = tag + '.npy'
    # print(seg_name)
    # print(img_name)
    # segs = [x for x in segs if tag in x]
    # print(len(segs))
    # segs.sort()
    # segs_stack = np.zeros((39, 224, 384)).astype('uint8')
    # for i in range(39):
    #     segs_stack[i] = np.load(os.path.join(seg_path,segs[i]))*255
    # seg = np.load(os.path.join(os.path.join(config.HOME_PATH, 'segmentations2d_cropped'),seg_name))*255
    # img = np.load(os.path.join(os.path.join(config.HOME_PATH, 'data_npy_resized_cropped'),img_name))
    # save_npy_vid(segs_stack, seg_name)
    # save_npy_vid(img, img_name)

if __name__ == '__main__':
    main()
