import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import tensorflow
import pydicom as di
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from PIL import Image
import config
import ants

HOME_PATH = '/home/jmh170430/Documents/mnt1/Advisory_Folder/Data/SRM_DCE_20DataSet/'
DATA_PATH = os.path.join(HOME_PATH, 'data/')
DATA_PATH_NPY = os.path.join(HOME_PATH, 'data_npy/')
TRAIN_PATH = os.path.join(HOME_PATH, 'Training/')
VAL_PATH = os.path.join(HOME_PATH, 'Validation/')
IMG_SIZE = 384

def printDirectory():
    PathDicom = DATA_PATH
    DCMFiles = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():
                DCMFiles.append(os.path.join(dirName,filename))

    print("Number of (.dcm) files =", len(DCMFiles))
    for k in DCMFiles:
        dimg = di.dcmread(k)
        pixel_array = dimg.pixel_array #shape (975, 400, 400)
        print(pixel_array.shape)

def dcmToNumpy():
    PathDicom = DATA_PATH
    DCMFiles = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():
                DCMFiles.append(os.path.join(dirName,filename))

    for k in DCMFiles:
        dimg = di.dcmread(k)
        pixel_array = dimg.pixel_array #shape (975, 400, 400)
        print(np.amax(pixel_array))
        print(pixel_array.dtype)
        num_slices = pixel_array.shape[0]//39 #25 slices
        pixel_array = pixel_array.astype('float')/np.amax(pixel_array)
        print(np.amax(pixel_array))
        print(pixel_array.dtype)
        pixel_array = (pixel_array*255).astype('uint8')
        pixel_array = np.array(pixel_array, dtype=np.uint8)
        print(np.amax(pixel_array))
        print(pixel_array.dtype)
        patient_num = str(k).replace('.dcm','')
        patient_num = patient_num.replace(DATA_PATH,'')
        patient_num = int(patient_num)
        print('Converting file ' + str(patient_num) + ' to npy...')
        print('File ' + str(patient_num) + ' has ' + str(num_slices) + ' slices')

        for i in range(num_slices):
            start_index = 39*i
            end_index = start_index+39

            img_name = 'Patient_'
            img_name+= str(patient_num).zfill(3) + '_Slice_'
            img_name+= str(i).zfill(3) + '.npy'


            pixel_subarray = pixel_array[start_index:end_index,:,:]
            np.save(os.path.join(DATA_PATH_NPY, img_name), pixel_subarray)
            print('Saved ' + img_name)

    print('Finished converting DICOM files to npy')

def resize_img(img):
    img_p = np.ndarray((39, IMG_SIZE, IMG_SIZE), dtype=np.uint16)
    for i in range(39):
        img_p[i] = cv2.resize(img[i], (IMG_SIZE, IMG_SIZE))

    return img_p

def splitResizeData():
    path = DATA_PATH_NPY
    n = os.listdir(path)
    n = [x for x  in n if x.endswith('.npy')]
    num_files = len(n)
    print('Number of files: ' + str(num_files))
    random.shuffle(n)
    img1 = np.load(os.path.join(path, n[0]))
    print('Data type: ', img1.dtype)
    print('Shape: ' + str(img1.shape))
    num_training = num_files*8//10

    print('Number of training files: ' + str(num_training))
    for name in n[0:num_training-1]:
        img = np.load(os.path.join(path, name))
        if(img.shape[1] != IMG_SIZE):
            img = resize_img(img)
        np.save(os.path.join(TRAIN_PATH, name), img)
    print('Finished saving training data')

    print('Number of validation files: ' + str(num_files-num_training))
    for name in n[num_training:]:
        img = np.load(os.path.join(path, name))
        if(img.shape[1] != IMG_SIZE):
            img = resize_img(img)
        np.save(os.path.join(VAL_PATH, name), img)
    print('Finished saving training data')

def generateImage():
    image_fixed = np.zeros((200,200),dtype=np.uint8)
    image_moving = np.zeros((200,200),dtype=np.uint8)
    image_fixed = cv2.rectangle(image_fixed, (120,50),(180,110),color=255,thickness=2)
    image_moving = cv2.rectangle(image_moving, (100,60),(150,110),color=255,thickness=2)
    fi = Image.fromarray(image_fixed)
    mi = Image.fromarray(image_moving)
    fi.save(os.path.join(config.IMAGES_PATH,'fixed_image1.png'))
    mi.save(os.path.join(config.IMAGES_PATH,'moving_image1.png'))

def downsample():
    fi = cv2.imread(os.path.join(config.IMAGES_PATH,'Patient_004_im003_renew.png'), cv2.IMREAD_GRAYSCALE)
    mi = cv2.imread(os.path.join(config.IMAGES_PATH,'Patient_004_im004_renew.png'), cv2.IMREAD_GRAYSCALE)
    thresh = 128
    fi = cv2.threshold(fi, thresh, 255, cv2.THRESH_BINARY)[1]
    mi = cv2.threshold(mi, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(os.path.join(config.IMAGES_PATH,'Patient_004_im003_renews.png'),fi)
    cv2.imwrite(os.path.join(config.IMAGES_PATH,'Patient_004_im004_renews.png'),mi)
    print(fi.shape)
    print(mi.shape)

def AffineReg():
    fi = ants.image_read(os.path.join(config.IMAGES_PATH,'Patient_004_im003_renews.png'))
    mi = ants.image_read(os.path.join(config.IMAGES_PATH,'Patient_004_im004_renews.png'))
    fi = fi[10:380,10:380]
    mi = mi[10:380,10:380]
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'AffineFast' )
    mygr = ants.create_warped_grid( mi )
    mywarpedgrid = ants.create_warped_grid( mi, grid_directions=(True,True), background=255, foreground=0, grid_step=10, grid_width=2,
                        transform=mytx['fwdtransforms'], fixed_reference_image=fi )
    print(mytx)
    warped_moving = mytx['warpedmovout']
    fi.plot(overlay=mi)
    fi.plot(overlay=warped_moving)
    ants.plot(mywarpedgrid)

def histogram():
    # dimg = di.dcmread(os.path.join(config.HOME_PATH, 'data/4.dcm'))
    # im = dimg.pixel_array #shape (975, 400, 400)
    im = np.load(os.path.join(config.DATA_PATH, 'Patient_004_Slice_012.npy'))
    print(im.shape)
    # im = np.load(os.path.join(config.HOME_PATH,'Voxelmorph/tutorial_data.npz'), 0)
    # im = im['train']
    print(im.dtype)
    im = im.astype('float')/255
    im = im**(1/1.5)

    print(np.amax(im))
    im = (im*255).astype(np.dtype('uint8'))
    # im = (im-51).astype('float')/(np.amax(im)-51)*255.astype(np.dtype('uint8'))
    print(np.amax(im))
    histo,bins = np.histogram(im, 256, [0,256])
    print(next((i for i, x in enumerate(histo[1:]) if x), None))
    # histo = cv2.calcHist([im],[0],None,[256], [0,256])
    #histo /= histo.sum()
    plt.plot(histo)
    plt.xlim([1,256])
    plt.ylim([0,0.12e6])
    plt.show()


if __name__ == '__main__':
    #printDirectory()
    #dcmToNumpy()
    #splitResizeData()
    #generateImage()
    #downsample()
    #AffineReg()
    #histogram()
