import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import tensorflow as tf
import numpy as np
import voxelmorph as vxm
import random
import h5py
import pickle
import cv2
import config
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input, concatenate, GlobalAveragePooling2D, Activation, Dropout, UpSampling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History

def mse_loss(y_true, y_pred):
    img_loss = tf.reduce_mean(tf.square(y_true[0] - y_pred[0]))
    seg_loss = tf.reduce_mean(tf.square(y_true[1] - y_pred[1]))
    return 0.8*img_loss + 0.2*seg_loss

def ncc_loss(y_true, y_pred):

    def incc_loss(static, moving):
        eps = tf.constant(1e-9, 'float32')

        static_mean = tf.reduce_mean(static, axis=[1, 2], keepdims=True)
        moving_mean = tf.reduce_mean(moving, axis=[1, 2], keepdims=True)
        # shape (N, 1, 1, C)

        static_std = tf.math.reduce_std(static, axis=[1, 2], keepdims=True)
        moving_std = tf.math.reduce_std(moving, axis=[1, 2], keepdims=True)
        # shape (N, 1, 1, C)

        static_hat = (static - static_mean)/(static_std + eps)
        moving_hat = (moving - moving_mean)/(moving_std + eps)
        # shape (N, H, W, C)

        ncc = tf.reduce_mean(static_hat * moving_hat)  # shape ()
        loss = -ncc
        return loss

    return 0.8*incc_loss(y_true[0], y_pred[0]) + 0.2*incc_loss(y_true[1], y_pred[1])

def dice_loss(y_true, y_pred, smooth=1.):

    def idice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 0.8*idice_loss(y_true[0], y_pred[0]) + 0.2*idice_loss(y_true[1], y_pred[1])

def grid_transform(theta, grid):
    # todo grid has nb
    nb = tf.shape(theta)[0]
    nh, nw, _ = tf.shape(grid)
    x = grid[..., 0]  # h,w
    y = grid[..., 1]

    x_flat = tf.reshape(x, shape=[-1])
    y_flat = tf.reshape(y, shape=[-1])
    ones = tf.ones_like(x_flat)
    grid_flat = tf.stack([x_flat, y_flat, ones])
    grid_flat = tf.expand_dims(grid_flat, axis=0)
    grid_flat = tf.tile(grid_flat, tf.stack([nb, 1, 1]))

    theta = tf.cast(theta, 'float32')
    grid_flat = tf.cast(grid_flat, 'float32')

    grid_new = tf.matmul(theta, grid_flat)  # n, 2, h*w
    grid_new = tf.transpose(grid_new, perm=[0,2,1])
    grid_new = tf.reshape(grid_new, [nb, nh, nw, 2])

    return grid_new

def grid_sample_2d(moving, grid):
    nb, nh, nw, nc = tf.shape(moving)

    x = grid[..., 0]  # shape (N, H, W)
    y = grid[..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # Scale x and y from [-1.0, 1.0] to [0, W] and [0, H] respectively.
    x = (x + 1.0) * 0.5 * tf.cast(nw-1, 'float32')
    y = (y + 1.0) * 0.5 * tf.cast(nh-1, 'float32')

    y_max = tf.cast(nh - 1, 'int32')
    x_max = tf.cast(nw - 1, 'int32')
    zero = tf.constant(0, 'int32')

    # The value at (x, y) is a weighted average of the values at the
    # four nearest integer locations: (x0, y0), (x1, y0), (x0, y1) and
    # (x1, y1) where x0 = floor(x), x1 = ceil(x).
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # Make sure indices are within the boundaries of the image.
    x0 = tf.clip_by_value(x0, zero, x_max)
    x1 = tf.clip_by_value(x1, zero, x_max)
    y0 = tf.clip_by_value(y0, zero, y_max)
    y1 = tf.clip_by_value(y1, zero, y_max)

    # Collect indices of the four corners.
    b = tf.ones_like(x0) * tf.reshape(tf.range(nb), [nb, 1, 1])
    idx_a = tf.stack([b, y0, x0], axis=-1)  # all top-left corners
    idx_b = tf.stack([b, y1, x0], axis=-1)  # all bottom-left corners
    idx_c = tf.stack([b, y0, x1], axis=-1)  # all top-right corners
    idx_d = tf.stack([b, y1, x1], axis=-1)  # all bottom-right corners
    # shape (N, H, W, 3)

    # Collect values at the corners.
    moving_a = tf.gather_nd(moving, idx_a)  # all top-left values
    moving_b = tf.gather_nd(moving, idx_b)  # all bottom-left values
    moving_c = tf.gather_nd(moving, idx_c)  # all top-right values
    moving_d = tf.gather_nd(moving, idx_d)  # all bottom-right values
    # shape (N, H, W, C)

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')

    # Calculate the weights.
    wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=-1)
    wb = tf.expand_dims((x1_f - x) * (y - y0_f), axis=-1)
    wc = tf.expand_dims((x - x0_f) * (y1_f - y), axis=-1)
    wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=-1)

    # Calculate the weighted sum.
    moved = tf.add_n([wa * moving_a, wb * moving_b, wc * moving_c,
                      wd * moving_d])
    return moved

def regular_grid_2d(height, width):
    x = tf.linspace(-1.0, 1.0, width)  # shape (W, )
    y = tf.linspace(-1.0, 1.0, height)  # shape (H, )

    X, Y = tf.meshgrid(x, y)  # shape (H, W), both X and Y

    grid = tf.stack([X, Y], axis=-1)
    return grid


def simple_cnn(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1), pretrained_weights=None):

    fixed_image = Input(shape=input_shape, name='fixed_image')
    moving_image = Input(shape=input_shape, name='moving_image')
    moving_seg = Input(shape=input_shape, name='moving_seg')
    #fixed_seg = Input(shape=input_shape, name='fixed_seg')
    x_in = concatenate([fixed_image, moving_image], axis=-1)
    # print(input_shape)

    # encoder
    x = Conv2D(32, kernel_size=3, strides=2, padding='same',
                      activation='relu')(x_in)            # 32 --> 16
    x = BatchNormalization()(x)                      # 16
    x = Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 # 16
    x = BatchNormalization()(x)                      # 16
    x = MaxPool2D(pool_size=2)(x)                    # 16 --> 8
    x = Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 # 8
    x = BatchNormalization()(x)                      # 8
    x = Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 # 8
    x = BatchNormalization()(x)                      # 8
    x = MaxPool2D(pool_size=2)(x)                    # 8 --> 4
    x = Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 # 4
    x = BatchNormalization()(x)                      # 4

    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)  # 4*4*32
    x = Dense(6, kernel_initializer='zeros',
                     bias_initializer=tf.constant_initializer([1,0,0,0,1,0]))(x)
    nb, _ = tf.shape(x)
    theta = tf.reshape(x, [nb, 2, 3])
    grid = regular_grid_2d(224, 384)
    grid_new = grid_transform(theta, grid)
    grid_new = tf.clip_by_value(grid_new, -1, 1)

    moved_image = grid_sample_2d(moving_image, grid_new)
    moved_seg = grid_sample_2d(moving_seg, grid_new)

    model = tf.keras.Model(inputs=[fixed_image, moving_image, moving_seg], outputs=[moved_image, moved_seg],
                           name='simple_cnn')

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    #model.summary()

    return model

# def grabFrameSegImage(patient_nb, slice_nb, frame_nb):
#     key = str(patient_nb) + '_' + str(slice_nb).zfill(2) + '_Object_2_' + str(frame_nb).zfill(4) + '.PNG'
#     img = cv2.imread(config.LABEL2D_PATH + key)
#     return img[...,0]

def getImageInfo(seg_filename):
    values = seg_filename.split('_')
    patient_nb = int(values[1].lstrip('0'))
    slice_nb = int(values[3].lstrip('0'))
    frame_nb = int(values[5].lstrip('0').replace('.npy',''))
    return [patient_nb, slice_nb, frame_nb]

def getImageName(patient_nb, slice_nb, frame_nb):
    return 'Patient_' + str(patient_nb).zfill(3) + '_Slice_' + str(slice_nb).zfill(3) + '_Frame_' + str(frame_nb).zfill(3) + '.npy'

def data_generator(directory, img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH, batch_size=config.BATCH_SIZE, data_path=config.TRAIN_PATH, seg_path=config.LABEL2D_TR_PATH):

    n = directory
    n = [x for x in n if 'Frame_001' not in x]
    random.shuffle(n)

    while True:

        moving_images = np.zeros((batch_size, img_height, img_width, 1)).astype('float')
        fixed_images = np.zeros((batch_size, img_height, img_width, 1)).astype('float')
        moving_segs = np.zeros((batch_size, img_height, img_width, 1)).astype('float')
        fixed_segs = np.zeros((batch_size, img_height, img_width, 1)).astype('float')

        idx = np.random.randint(0, len(n), size=batch_size) #Random volumes indexes

        for i in range(batch_size):

            moving_info = getImageInfo(n[idx[i]])
            fixed_info = [moving_info[0], moving_info[1], moving_info[2]-1]

            moving_image = np.load(os.path.join(data_path, getImageName(*moving_info)))
            fixed_image = np.load(os.path.join(data_path, getImageName(*fixed_info)))
            moving_seg = np.load(os.path.join(seg_path, 'Seg_' + getImageName(*moving_info)))
            fixed_seg = np.load(os.path.join(seg_path, 'Seg_' + getImageName(*fixed_info)))

            if(config.TO_AUGMENT):

                # # Random flip
                # toFlip = random.random()
                # if toFlip<=0.25:
                #     flipCode=0
                # elif toFlip<=0.5 and toFlip>0.25:
                #     flipCode=-1
                # elif toFlip>0.5 and toFlip<=75:
                #     flipCode=1
                # else:
                #     flipCode=2
                #
                # if flipCode!=2:
                #     moving_image = cv2.flip(moving_image, flipCode)
                #     fixed_image = cv2.flip(fixed_image, flipCode)

                # Random translation
                x_trans1 = random.randint(-5,5)
                y_trans1 = random.randint(-5,5)
                x_trans2 = random.randint(-5,5)
                y_trans2 = random.randint(-5,5)
                translation_mat1 = np.float32([ [1,0,x_trans1], [0,1,y_trans1] ])
                translation_mat2 = np.float32([ [1,0,x_trans2], [0,1,y_trans2] ])
                moving_image = cv2.warpAffine(moving_image, translation_mat1, (img_width, img_height))
                fixed_image = cv2.warpAffine(fixed_image, translation_mat2, (img_width, img_height))
                moving_seg = cv2.warpAffine(moving_seg, translation_mat1, (img_width, img_height))
                fixed_seg = cv2.warpAffine(fixed_seg, translation_mat2, (img_width, img_height))

                #Random ortation
                # degrees = random.randint(0,3) * 90
                # if degrees != 0:
                #     if degrees == 90:
                #         moving_image = cv2.rotate(moving_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                #         fixed_image = cv2.rotate(fixed_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                #     elif degrees == 180:
                #         moving_image = cv2.rotate(moving_image, cv2.ROTATE_180)
                #         fixed_image = cv2.rotate(fixed_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                #     else:
                #         moving_image = cv2.rotate(moving_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                #         fixed_image = cv2.rotate(fixed_image, cv2.cv2.ROTATE_90_CLOCKWISE)

                # Add channel
                moving_image = moving_image[...,np.newaxis]
                fixed_image = fixed_image[...,np.newaxis]
                moving_seg = moving_seg[...,np.newaxis]
                fixed_seg = fixed_seg[...,np.newaxis]



                # Random rotation
                # rg1 = random.randint(-15,15)
                # rg2 = random.randint(-15,15)
                # moving_image = tf.keras.preprocessing.image.random_rotation(moving_image, rg1, row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest',interpolation_order=1)
                # fixed_image = tf.keras.preprocessing.image.random_rotation(fixed_image, rg2, row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest',interpolation_order=1)

            else:
                moving_image = moving_image[...,np.newaxis]
                fixed_image = fixed_image[...,np.newaxis]
                moving_seg = moving_seg[...,np.newaxis]
                fixed_seg = fixed_seg[...,np.newaxis]

            moving_image = moving_image.astype('float')/255.
            fixed_image = fixed_image.astype('float')/255.
            moving_image = moving_image.astype('float')
            fixed_image = fixed_image.astype('float')

            # moving_image = show_edges((moving_image*255).astype(np.dtype('uint8')))
            # fixed_image = show_edges((fixed_image*255).astype(np.dtype('uint8')))

            moving_images[i] = moving_image#.astype('float')/255
            fixed_images[i] = fixed_image#.astype('float')/255
            moving_segs[i] = moving_seg#.astype('float')/255
            fixed_segs[i] = fixed_seg#.astype('float')/255

        # inputs = [moving_images, fixed_images]
        # outputs = [fixed_images, zero_phi]
        inputs = [fixed_images, moving_images, moving_segs]
        outputs = [fixed_images, fixed_segs]

        yield inputs, outputs



def main():
    from sklearn.model_selection import train_test_split
    dir = os.listdir(config.TRAIN_PATH)
    # x_train, x_val = train_test_split(dir, train_size=0.9)
    # train_generator = data_generator(x_train)
    train_generator = data_generator(dir)
    #split_resize_data()
    model = simple_cnn()
    if(config.USING_PRETRAINED_WEIGHTS):
        model = simple_cnn(pretrained_weights = config.WEIGHTS_PATH)

    losses = [vxm.losses.MSE().loss, vxm.losses.MSE().loss]
    loss_weights = [config.ALPHA_PARAM, 0]

    opt = Adam(lr=1E-4)
    model.compile(loss=losses, loss_weights=loss_weights,
                optimizer=opt)
    checkpoint = ModelCheckpoint(config.WEIGHTS_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', verbose=1, patience=config.PATIENCE, mode='min')
    callbacks_list = [checkpoint, earlystopping]
    history=model.fit(train_generator, epochs=config.NB_EPOCHS,
                    steps_per_epoch=config.STEPS_PER_EPOCH,
                    callbacks=callbacks_list,
                    verbose=1)
    with open(config.HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    main()
