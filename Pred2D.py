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
import Eval
import voxelmorph as vxm
import neurite as ne
import ants
from PIL import Image
import datetime

from Train2D_P import get_vxm2D, vxm_data_generator, show_edges, adjust_gamma
from Train2DwSeg import simple_cnn, mse_loss, ncc_loss, dice_loss, data_generator, getImageInfo, getImageName
from Deform2DwSeg import deform_cnn

IMG_SCALE = 715. #1436

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def my_data_generator(data_path, img_size=config.IMG_SIZE, patient_name=config.PATIENT_NAME, ndims=2, slice_nb=12, frame_nb=1):
    n = os.listdir(data_path)
    n = [x for x in n if patient_name in x]
    n.sort()
    batch_size = 1
    zero_phi = np.zeros([batch_size, img_size, img_size, ndims])
    vol = np.load(os.path.join(data_path, n[slice_nb]))
    print(vol.shape)

    while True:

        moving_images = np.zeros((1, img_size, img_size, 1)).astype('float')
        fixed_images = np.zeros((1, img_size, img_size, 1)).astype('float')

        moving_image = vol[frame_nb,...]
        fixed_image = vol[frame_nb-1, ...]

        gamma = 1.5
        moving_image = (moving_image.astype('float')/255)**(1.0/gamma)
        fixed_image = (fixed_image.astype('float')/255)**(1.0/gamma)
        print(moving_image.shape)

        moving_image = moving_image[...,np.newaxis]
        fixed_image = fixed_image[...,np.newaxis]

        moving_images[0] = moving_image#.astype('float')/255
        fixed_images[0] = fixed_image#.astype('float')/255

        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)


def predict_test_vxm():
    npz = np.load(config.VXM_DATA_PATH)
    x_train = npz['train']
    x_val = npz['validate']

    # the 208 volumes are of size 160x192
    vol_shape = x_train.shape[1:]
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    print('train shape:', x_train.shape)
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
    # train_generator = vxm_data_generator(x_train, batch_size=8)
    # in_sample, out_sample = next(train_generator)
    vxm_model.load_weights(config.WEIGHTS_PATH)
    val_generator = vxm_data_generator(x_val, batch_size = 1)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)
    images = [img[0, :, :, 0] for img in val_input + val_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

def predict_test_my():

    # the 208 volumes are of size 160x192
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    vxm_model = vxm.networks.VxmDense((384,384), nb_features, int_steps=0)
    vxm_model.load_weights(config.WEIGHTS_PATH)
    val_generator = my_data_generator(config.TRAIN_PATH, frame_nb=20)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)
    images = [img[0, :, :, 0] for img in val_input + val_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);
    flow = val_pred[1].squeeze()[::3,::3]
    ne.plot.flow([flow], width=5);

def normalize(array):
    array = (array-np.amin(array))/(np.amax(array)-np.amin(array))
    return array

def predict():
    # vxm_model = get_vxm2D(pretrained_weights=config.WEIGHTS_PATH)
    # vxm_model.summary()
    test_gen = test_data_generator(data_path=config.TRAIN_PATH)
    orig_inputs, __ = next(test_gen)

    test_generator = test_data_generator_consec(data_path=config.TRAIN_PATH, first=True)
    test_inputs, __ = next(test_generator)
    inshape = test_inputs[0].shape[1:-1]
    nb_features = [
        [32, 32, 32, 32],
        [32, 32, 32, 32, 32, 16]
    ]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(config.WEIGHTS_PATH)
    # trans_model = vxm.networks.Transform(inshape)

    # plot_inputs_edges(test_inputs)
    test_preds = vxm_model.predict(test_inputs)
    # warp = vxm_model.register(test_inputs[0], test_inputs[1])
    # moved = trans_model.predict([test_inputs[0], warp])
    #
    # images = [img[0, :, :, 0] for img in test_inputs + moved + warp]
    # titles = ['moving', 'fixed', 'moved', 'flow']
    # ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

    # print('length: ', len(test_preds))
    # print('inputs',test_inputs[1].shape)
    # print('preds',test_preds[0].shape)
    # print('maps',test_preds[1].shape)
    # preds_arr = np.zeros((39, config.IMG_SIZE, config.IMG_SIZE, 1)).astype('float')
    # flow_arr = np.zeros((39, config.IMG_SIZE, config.IMG_SIZE)).astype('float')
    # preds_arr[0] = test_inputs[1][0] #Pred 0
    # preds_arr[1] = test_preds[0][0] #Pred 1
    # flow_arr[1] = test_preds[1][0, :, :, 0]

    images = [img[0, :, :, 0] for img in test_inputs + test_preds]
    orig_arr = np.squeeze(orig_inputs[1])
    preds_arr = np.zeros((39, config.IMG_SIZE, config.IMG_SIZE)).astype('float')
    flow_arr = np.zeros((39, config.IMG_SIZE, config.IMG_SIZE)).astype('float')
    diff_arr = np.zeros((39, config.IMG_SIZE, config.IMG_SIZE)).astype('float')
    preds_arr[0] = orig_arr[0]
    preds_arr[1] = images[2]
    flow_arr[1] = images[3]
    diff_arr[0] = np.absolute(np.subtract(preds_arr[0], orig_arr[0]))

    for i in range(2, 39):
        test_generator = test_data_generator_consec(data_path=config.TRAIN_PATH, fixed_image=preds_arr[i-1], frame_nb=i)
        test_inputs, __ = next(test_generator)
        test_preds = vxm_model.predict(test_inputs)
        images = [img[0, :, :, 0] for img in test_inputs + test_preds]
        preds_arr[i] = images[2]
        flow_arr[i] = images[3]
        diff_arr[i] = np.absolute(np.subtract(preds_arr[i],orig_arr[i]))
    # for i in range(2, 39):
    #     test_generator = test_data_generator_consec(data_path=config.TRAIN_PATH, fixed_image=preds_arr[i-1], frame_nb=i)
    #     test_inputs, __ = next(test_generator)
    #     warp = vxm_model.register(test_inputs[0], test_inputs[1])
    #     moved = trans_model.predict([test_inputs[0], warp])
    #     preds_arr[i] = moved[0]
    #     flow_arr[i] = warp[0, :, :, 0]
    min_flow = np.amin(flow_arr)
    max_flow = np.amax(flow_arr)
    flow_arr = normalize(flow_arr)
    print(min_flow)
    print(max_flow)
    plot_images(orig_arr, preds_arr, diff_arr, flow_arr, min_flow, max_flow)
    save_images(orig_arr, mode='orig')
    save_images(preds_arr, mode='pred')

    save_video(preds_arr, mode='pred')
    save_video(orig_arr, mode='orig')
    save_video(flow_arr, mode='map')
    save_video(diff_arr, mode='diff')

    # plot_inputs_edges(orig_inputs)


def predict_affine_seg():
    dir = os.listdir(config.TEST_PATH)
    test_gen = test_data_generator_consec(data_path=config.TEST_PATH, seg_path=config.LABEL2D_TS_PATH)
    inputs, __ = next(test_gen)
    model = simple_cnn(pretrained_weights=config.WEIGHTS_PATH)
    outputs = model.predict(inputs)
    fixed_images = inputs[0]
    moving_images = inputs[1]
    moving_segs = inputs[2]
    moved_images = outputs[0]
    moved_segs = outputs[1]
    plot_images_affine(moving_images, moved_images, fixed_images, moving_segs, moved_segs)
    moved_images = np.squeeze(moved_images)
    moving_images = np.squeeze(moving_images)
    fixed_images = np.squeeze(fixed_images)
    moving_segs = np.squeeze(moving_segs)
    moved_segs = np.squeeze(moved_segs)

    if(config.TO_EVAL):
        evaluate_images(moving_images, moved_images, fixed_images, moving_segs, moved_segs)

def predict_affine_consec():
    model = simple_cnn(pretrained_weights=os.path.join(config.HOME_PATH, 'Weights/Seg2D_Affine_1_18.h5'))

    fixed_images = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    moving_images = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    moving_segs = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    reg_images = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    reg_segs = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')

    begin_time = datetime.datetime.now()

    test_gen_first = test_data_generator_consec(data_path=config.TEST_PATH, seg_path=config.LABEL2D_TS_PATH, first=True)
    moving, __ = next(test_gen_first)
    registered = model.predict(moving)
    fixed_images[0] = np.squeeze(moving[0])
    moving_images[0] = np.squeeze(moving[1])
    moving_segs[0] = np.squeeze(moving[2])
    reg_images[0] = np.squeeze(registered[0])
    reg_segs[0] = np.squeeze(registered[1])

    for i in range(2,38):
        test_gen = test_data_generator_consec(data_path=config.TEST_PATH, seg_path=config.LABEL2D_TS_PATH, fixed_image=reg_images[i-2], fixed_seg=reg_segs[i-2], frame_nb=i, first=False)
        moving, __ = next(test_gen)
        registered = model.predict(moving)
        fixed_images[i-1] = np.squeeze(moving[0])
        moving_images[i-1] = np.squeeze(moving[1])
        moving_segs[i-1] = np.squeeze(moving[2])
        reg_images[i-1] = np.squeeze(registered[0])
        reg_segs[i-1] = np.squeeze(registered[1])

    print(datetime.datetime.now()-begin_time)

    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_fixed_images.npy'), fixed_images)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_moving_images.npy'), moving_images)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_moving_segs.npy'), moving_segs)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_images.npy'), reg_images)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_segs.npy'), reg_segs)

def getSeg(patient_nb,slice_nb,frame_nb):
    name = 'Seg_' + getImageName(patient_nb,slice_nb,frame_nb)
    fixed_seg = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH)).astype('float')

    fixed_seg = np.load(os.path.join(config.LABEL2D_TS_PATH, name))

    fixed_seg = fixed_seg.astype('float')
    return fixed_seg

def predict_deform_consec():
    model = deform_cnn(pretrained_weights=config.WEIGHTS_PATH)

    fixed_images = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_fixed_images.npy'))
    moving_images = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_images.npy'))
    moving_segs = np.load(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_affine_segs.npy'))
    defreg_images = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    defreg_segs = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    flow_images = np.zeros((38,config.IMG_HEIGHT,config.IMG_WIDTH)).astype('float')
    first_seg = getSeg(*getImageInfo(config.PATIENT_NAME + '_Frame_001'))

    begin_time = datetime.datetime.now()

    test_gen_first = def_data_generator_consec(data_path=config.TEST_PATH, seg_path=config.LABEL2D_TS_PATH, moving_image=moving_images[0], moving_seg=moving_segs[0], fixed_image=fixed_images[0], fixed_seg=first_seg, frame_nb=1, first=False)
    moving, __ = next(test_gen_first)
    registered = model.predict(moving)
    defreg_images[0] = np.squeeze(registered[0])
    defreg_segs[0] = np.squeeze(registered[1])
    flow_images[0] = registered[2][0,:,:,0]

    for i in range(2,38):
        test_gen = def_data_generator_consec(data_path=config.TEST_PATH, seg_path=config.LABEL2D_TS_PATH, moving_image=moving_images[i-1], moving_seg=moving_segs[i-1],fixed_image=defreg_images[i-2], fixed_seg=defreg_segs[i-2], frame_nb=i, first=False)
        moving, __ = next(test_gen)
        registered = model.predict(moving)
        defreg_images[i-1] = np.squeeze(registered[0])
        defreg_segs[i-1] = np.squeeze(registered[1])
        flow_images[i-1] = registered[2][0,:,:,0]

    flow_images_norm = normalize(flow_images)

    print(datetime.datetime.now()-begin_time)

    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_deformed_images.npy'), defreg_images)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_deformed_segs.npy'), defreg_segs)
    # np.save(os.path.join(config.OUTPUTS_PATH, config.PATIENT_NAME + '_flow_images_norm.npy'), flow_images_norm)



def test_data_generator_consec(data_path, seg_path, img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH, patient_name=config.PATIENT_NAME, ndims=2, fixed_image=None, fixed_seg=None, frame_nb=1, first=False):
    n = os.listdir(data_path)
    n = [x for x in n if patient_name in x]
    n.sort()
    batch_size = 1 #Can only be 39 or less

    while True:

        moving_images = np.zeros((1, img_height, img_width, 1)).astype('float')
        fixed_images = np.zeros((1, img_height, img_width, 1)).astype('float')
        moving_segs = np.zeros((1, img_height, img_width, 1)).astype('float')
        fixed_segs = np.zeros((1, img_height, img_width, 1)).astype('float')
        zero_phi = np.zeros([1, img_height, img_width, 2])

        if(first):
            moving_info = getImageInfo(n[1])
            fixed_info = [moving_info[0], moving_info[1], moving_info[2]-1]

            moving_image = np.load(os.path.join(data_path, getImageName(*moving_info)))
            fixed_image = np.load(os.path.join(data_path, getImageName(*fixed_info)))
            moving_seg = np.load(os.path.join(seg_path, 'Seg_' + getImageName(*moving_info)))
            fixed_seg = np.load(os.path.join(seg_path, 'Seg_' + getImageName(*fixed_info)))

            moving_image = moving_image[...,np.newaxis]
            fixed_image = fixed_image[...,np.newaxis]
            moving_seg = moving_seg[...,np.newaxis]
            fixed_seg = fixed_seg[...,np.newaxis]

            moving_image = moving_image.astype('float')/255.
            fixed_image = fixed_image.astype('float')/255.
            moving_image = moving_image.astype('float')
            fixed_seg = fixed_seg.astype('float')

        else:
            moving_info = getImageInfo(n[frame_nb])

            moving_image = np.load(os.path.join(data_path, getImageName(*moving_info)))
            moving_seg = np.load(os.path.join(seg_path, 'Seg_' + getImageName(*moving_info)))

            moving_image = moving_image[...,np.newaxis]
            fixed_image = fixed_image[...,np.newaxis]
            moving_seg = moving_seg[...,np.newaxis]
            fixed_seg = fixed_seg[...,np.newaxis]

            moving_image = moving_image.astype('float')/255.
            moving_image = moving_image.astype('float')

            # moving_image = show_edges((moving_image*255).astype(np.dtype('uint8')))
            # fixed_image = show_edges((fixed_image*255).astype(np.dtype('uint8')))

        moving_images[0] = moving_image#.astype('float')/255
        fixed_images[0] = fixed_image#.astype('float')/255
        moving_segs[0] = moving_seg#.astype('float')/255
        fixed_segs[0] = fixed_seg#.astype('float')/255

        # inputs = [moving_images, fixed_images]
        # outputs = [fixed_images, zero_phi]
        inputs = [fixed_images, moving_images, moving_segs]
        outputs = [fixed_images, fixed_segs, zero_phi]


        yield inputs, outputs

def def_data_generator_consec(data_path, seg_path, img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH, patient_name=config.PATIENT_NAME, ndims=2, moving_image=None, moving_seg=None, fixed_image=None, fixed_seg=None, frame_nb=1, first=False):
    batch_size = 1 #Can only be 39 or less

    while True:

        moving_images = np.zeros((1, img_height, img_width, 1)).astype('float')
        fixed_images = np.zeros((1, img_height, img_width, 1)).astype('float')
        moving_segs = np.zeros((1, img_height, img_width, 1)).astype('float')
        fixed_segs = np.zeros((1, img_height, img_width, 1)).astype('float')

        moving_image = moving_image[...,np.newaxis]
        fixed_image = fixed_image[...,np.newaxis]
        moving_seg = moving_seg[...,np.newaxis]
        fixed_seg = fixed_seg[...,np.newaxis]


            # moving_image = show_edges((moving_image*255).astype(np.dtype('uint8')))
            # fixed_image = show_edges((fixed_image*255).astype(np.dtype('uint8')))

        moving_images[0] = moving_image#.astype('float')/255
        fixed_images[0] = fixed_image#.astype('float')/255
        moving_segs[0] = moving_seg#.astype('float')/255
        fixed_segs[0] = fixed_seg#.astype('float')/255

        # inputs = [moving_images, fixed_images]
        # outputs = [fixed_images, zero_phi]
        inputs = [fixed_images, moving_images, moving_segs]
        outputs = [fixed_images, fixed_segs]


        yield inputs, outputs



def save_slice_video(inputs, preds, mode='input'):
    nametag = '_orig'
    n = 0
    if(mode=='pred'):
        nametag = '_pred'
        n = 2
    elif(mode=='targ'):
        nametag = '_targ'
        n = 1
    elif(mode=='map'):
        nametag = '_map'
        n = 3

    images = [img[:,:,:,0] for img in inputs + preds]
    out = cv2.VideoWriter(os.path.join(config.PRED_PATH,config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB)+nametag+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (config.IMG_SIZE,config.IMG_SIZE), False)

    for i in range(len(images[0])):
        image = images[n][i]
        image = (image*255).astype(np.dtype('uint8'))
        out.write(image)
    out.release()

def save_video(array, mode='input'):
    array = np.squeeze(array)
    print(array.shape)
    nametag = '_orig'
    if(mode=='pred'):
        nametag = '_pred'
    elif(mode=='targ'):
        nametag = '_targ'
    elif(mode=='map'):
        nametag = '_map'

    elif(mode=='diff'):
        nametag = '_diff'

    out = cv2.VideoWriter(os.path.join (config.PRED_PATH,config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB)+nametag+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10,
            (config.IMG_SIZE,config.IMG_SIZE), False)

    for i in range(len(array)):
        image = array[i]
        image = (image*255).astype(np.dtype('uint8'))
        out.write(image)
    out.release()

def plot_images(orig, pred, diff, flow, flowmin, flowmax):
    images = np.stack((orig, pred, diff, flow), axis=0)
    #lin = np.linspace(np.ceil(flowmin), np.floor(flowmax), np.floor(flowmax)-np.ceil(flowmin)+1)
    # flow *= (flowmax - flowmin)
    # flow += flowmin
    f, axarr = plt.subplots(1,5)
    for i in range(len(images[0])):
        plt.clf()
        axarr[0].imshow(images[0][i], cmap='gray', vmin=0, vmax=1)
        axarr[0].set_title('Original')
        axarr[0].axis('off')
        axarr[1].imshow(images[1][i-1], cmap='gray', vmin=0, vmax=1)
        axarr[1].set_title('Reference')
        axarr[1].axis('off')
        axarr[2].imshow(images[1][i], cmap='gray', vmin=0, vmax=1)
        axarr[2].set_title('Registered')
        axarr[2].axis('off')
        axarr[3].imshow(images[2][i], cmap='gray', vmin=0, vmax=1)
        axarr[3].set_title('Difference')
        axarr[3].axis('off')
        axarr[4].imshow(images[3][i], cmap='gray', vmin=0, vmax=1)
        axarr[4].set_title('Deformation')
        axarr[4].axis('off')
        # cbar = f.colorbar(flow_im, ax=axarr[1, 1], ticks=[flowmin, 0, flowmax])
        # cbar.ax.set_yticklabels([str(round(flowmin,2)), '0', str(round(flowmax,2))])
        f.savefig(os.path.join(config.PRED_PATH, config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB).zfill(3)+'_Frame_'+str(i).zfill(3)+'.png'),bbox_inches='tight')

def plot_images_affine(moving_images, moved_images, fixed_images, moving_segs, moved_segs):
    images = np.stack((moving_images, moved_images, fixed_images, moving_segs, moved_segs), axis=0)

    for i in range(len(images[0])):
        plt.clf()
        f, axarr = plt.subplots(2,3)
        axarr[0,0].imshow(images[0][i], cmap='gray', vmin=0, vmax=1)
        axarr[0,0].set_title('Moving')
        axarr[0,0].axis('off')
        axarr[0,1].imshow(images[1][i], cmap='gray', vmin=0, vmax=1)
        axarr[0,1].set_title('Moved')
        axarr[0,1].axis('off')
        axarr[0,2].imshow(images[2][i], cmap='gray', vmin=0, vmax=1)
        axarr[0,2].set_title('Fixed')
        axarr[0,2].axis('off')
        axarr[1,0].imshow(images[3][i], cmap='gray', vmin=0, vmax=1)
        axarr[1,0].set_title('Moving Seg')
        axarr[1,0].axis('off')
        axarr[1,1].imshow(images[4][i], cmap='gray', vmin=0, vmax=1)
        axarr[1,1].set_title('Moved Seg')
        axarr[1,1].axis('off')
        # cbar = f.colorbar(flow_im, ax=axarr[1, 1], ticks=[flowmin, 0, flowmax])
        # cbar.ax.set_yticklabels([str(round(flowmin,2)), '0', str(round(flowmax,2))])
        #f.savefig(os.path.join(config.PRED_PATH, config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB).zfill(3)+'_Frame_'+str(i).zfill(3)+'.png'),bbox_inches='tight')
        f.savefig(os.path.join(config.OUTPUTS_PATH, 'Affine_Trial_'+ str(i)+'.png'),bbox_inches='tight')

def evaluate_images(moving_images, moved_images, fixed_images, moving_segs, moved_segs):

    def plot_eval_images(m2f_eval, p2f_eval, nTest, mode='none'):
        plt.clf()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(m2f_eval, cmap='gray', vmin=0, vmax=255)
        axarr[0].set_title('Moving to Fixed Comparison')
        axarr[0].axis('off')
        axarr[1].imshow(p2f_eval, cmap='gray', vmin=0, vmax=255)
        axarr[1].set_title('Registered to Fixed Comparison')
        axarr[1].axis('off')
        f.savefig(os.path.join(config.OUTPUTS_PATH, 'Affine_Test_wAugment_'+ str(nTest).zfill(2) + str(mode) + '.png'),bbox_inches='tight')
        plt.close()

    def plot_single_image(eval, nTest, mode='none'):
        plt.clf()
        plt.imshow(eval)
        plt.savefig(os.path.join(config.OUTPUTS_PATH, 'Affine_Test_wAugment_'+ str(nTest).zfill(2) + str(mode) + '.png'),bbox_inches='tight')

    moved_images = (moved_images*255).astype(np.dtype('uint8'))
    moving_images = (moving_images*255).astype(np.dtype('uint8'))
    fixed_images = (fixed_images*255).astype(np.dtype('uint8'))
    moving_segs = (moving_segs*255).astype(np.dtype('uint8'))
    moved_segs = (moved_segs*255).astype(np.dtype('uint8'))
    for i in range(len(moving_images)):
        alpha1 = Eval.alpha_blend(moving_images[i], fixed_images[i])
        alpha2 = Eval.alpha_blend(moved_images[i], fixed_images[i])
        checker1 = Eval.checkerboard(moving_images[i], fixed_images[i], 8)
        checker2 = Eval.checkerboard(moved_images[i], fixed_images[i], 8)
        subtract1 = Eval.imsubtract(moving_images[i], fixed_images[i])
        subtract2 = Eval.imsubtract(moved_images[i], fixed_images[i])
        #Needs fixing
        borderoverlay = Eval.boundary_overlay(fixed_images[i], moving_segs[i], moved_segs[i])
        #End needs fixing
        plot_eval_images(alpha1, alpha2, i, mode='alpha')
        plot_eval_images(checker1, checker2, i, mode='checker')
        plot_eval_images(normalize(subtract1), normalize(subtract2), i, mode='subtract')
        plot_single_image(borderoverlay, i, mode='borderoverlay')


def save_images(array, mode='input'):
    array = np.squeeze(array)
    print(array.shape)
    nametag = '_orig'
    if(mode=='pred'):
        nametag = '_pred'
    elif(mode=='targ'):
        nametag = '_targ'
    elif(mode=='map'):
        nametag = '_map'

    elif(mode=='diff'):
        nametag = '_diff'


    for i in range(len(array)):
        image = array[i]
        image = (image*255).astype(np.dtype('uint8'))
        im = Image.fromarray(image)
        im.save(os.path.join(config.PRED_PATH,config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB)+'Frame_'+str(i)+nametag+'.png'))



def plot_inputs_edges(inputs):

    moving_8bit = (inputs[0]*255).astype(np.dtype('uint8'))
    # fixed_8bit = (inputs[1]*255).astype(np.dtype('uint8'))
    # fixed_edges = show_edges(fixed_8bit)

    for i in range(len(moving_8bit)):
        moving_edges = cv2.Canny(moving_8bit[i,...,0],50, 100)
        plt.clf()
        plt.imshow(moving_edges, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        # f, axarr = plt.subplots(1,2)
        # axarr[0].imshow(moving_8bit[i,...,0], cmap='gray', vmin=0, vmax=255)
        # axarr[0].set_title('Moving')
        # axarr[0].axis('off')
        # axarr[1].imshow(fixed_8bit[i,...,0], cmap='gray', vmin=0, vmax=255)
        # axarr[1].set_title('Fixed')
        # axarr[1].axis('off')
        # axarr[1,0].imshow(cv2.add(moving_edges[i,...,0],moving_8bit[i,...,0]), cmap='gray', vmin=0, vmax=255)
        # axarr[1,0].set_title('Moving input')
        # axarr[1,0].axis('off')
        # axarr[1,1].imshow(cv2.add(fixed_edges[i,...,0],fixed_8bit[i,...,0]), cmap='gray', vmin=0, vmax=255)
        # axarr[1,1].set_title('Fixed input')
        # axarr[1,1].axis('off')
        # plt.savefig(os.path.join(config.PRED_PATH, config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB).zfill(3)+'_Frame_'+str(i).zfill(3)+'_Gamma.png'),bbox_inches='tight')
        plt.savefig(os.path.join(config.IMAGES_PATH, config.PATIENT_NAME+'_im'+str(i).zfill(3)+'.png'),bbox_inches='tight')

# def cropped_slice_vid(patient_name=config.PATIENT_NAME, config.SLICE_NB=config.SLICE_NB):
#     file_name = patient_name + '_Slice_' + str(config.SLICE_NB).zfill(3) + '.npy'
#     inputs = np.load(os.path.join(config.TRAIN_PATH, file_name))
#     out = cv2.VideoWriter(os.path.join(config.PRED_PATH,config.PATIENT_NAME+'_Slice_'+str(config.SLICE_NB)+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (config.IMG_SIZE,config.IMG_SIZE), False)
#     for i in range(39):
#         image = inputs[i]
#         image = (image*255).astype(np.dtype('uint8'))
#         out.write(image)
#     out.release()



def plot_hist():
    history = pickle.load(open(config.HISTORY_PATH, "rb"))
    plt.clf()

    plt.clf()
    plt.plot(history['mse_loss'])
    plt.plot(history['val_mse_loss'])
    plt.title('Model MSE loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(config.PLOTS_PATH, 'flow_loss2D_P4.png'), bbox_inches='tight')
    #summarize history for loss
    plt.clf()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(config.PLOTS_PATH, 'loss2D_Affine.png'), bbox_inches='tight')

if __name__ == '__main__':
    #plot_hist()
    #predict_test_my()
    #predict_affine()
    #predict_affine_consec()
    predict_deform_consec()
    #cropped_slice_vid()
