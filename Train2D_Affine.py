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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input, concatenate, GlobalAveragePooling2D, Activation, Dropout, UpSampling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History

def mse_loss(static, moving):
    loss = tf.reduce_mean(tf.square(moving - static))
    return loss

def ncc_loss(static, moving):
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


def conv2D_MaxPooling_Layer(input, nfilters):
    out = Conv2D(nfilters, kernel_size=3, strides=1, padding='same', activation='relu')(input)
    out = BatchNormalization()(out)
    out = MaxPool2D(pool_size=2)(out)
    return out

def simple_cnn(input_shape=(384, 384), pretrained_weights=None):

    in_channels = 1
    # out_channels = 3
    input_shape = input_shape + (in_channels,)
    moving = Input(shape=input_shape, name='moving')
    static = Input(shape=input_shape, name='static')
    x_in = concatenate([static, moving], axis=-1)
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
    grid = regular_grid_2d(384, 384)
    grid_new = grid_transform(theta, grid)
    grid_new = tf.clip_by_value(grid_new, -1, 1)

    moved = grid_sample_2d(moving, grid_new)

    model = tf.keras.Model(inputs=[static, moving], outputs=moved,
                           name='simple_cnn')

    if(pretrained_weights):
        model.load_weights(pretrained_weights)


    return model

def grabFrameSegImage(patient_nb, slice_nb, frame_nb):
    key = str(patient_nb) + '_' + str(slice_nb).zfill(2) + '_Object_2_' + str(frame_nb).zfill(4) + '.PNG'
    img = cv2.imread(config.LABEL2D_PATH + key)
    return img[...,0]

def data_generator(directory, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, data_path=config.TRAIN_PATH):

    n = directory
    random.shuffle(n)

    while True:

        moving_images = np.zeros((batch_size, img_size, img_size, 1)).astype('float')
        fixed_images = np.zeros((batch_size, img_size, img_size, 1)).astype('float')

        idx = np.random.randint(0, len(n), size=batch_size) #Random volumes indexes

        for i in range(batch_size):
            vol = np.load(os.path.join(data_path, n[idx[i]]))
            rindex = random.randint(1,38)
            moving_image = vol[rindex,...] #Grab a random frame
            fixed_image = vol[rindex-1,...]

            if(config.TO_AUGMENT):

                # Random flip
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
                x_trans1 = random.randint(-10,10)
                y_trans1 = random.randint(-10,10)
                x_trans2 = random.randint(-10,10)
                y_trans2 = random.randint(-10,10)
                translation_mat1 = np.float32([ [1,0,x_trans1], [0,1,y_trans1] ])
                translation_mat2 = np.float32([ [1,0,x_trans2], [0,1,y_trans2] ])
                moving_image = cv2.warpAffine(moving_image, translation_mat1, (img_size, img_size))
                fixed_image = cv2.warpAffine(fixed_image, translation_mat2, (img_size, img_size))

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



                # Random rotation
                # rg1 = random.randint(-15,15)
                # rg2 = random.randint(-15,15)
                # moving_image = tf.keras.preprocessing.image.random_rotation(moving_image, rg1, row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest',interpolation_order=1)
                # fixed_image = tf.keras.preprocessing.image.random_rotation(fixed_image, rg2, row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest',interpolation_order=1)

            else:
                moving_image = moving_image[...,np.newaxis]
                fixed_image = fixed_image[...,np.newaxis]

            moving_image = moving_image.astype('float')/255.
            fixed_image = fixed_image.astype('float')/255.

            # moving_image = show_edges((moving_image*255).astype(np.dtype('uint8')))
            # fixed_image = show_edges((fixed_image*255).astype(np.dtype('uint8')))

            moving_images[i] = moving_image#.astype('float')/255
            fixed_images[i] = fixed_image#.astype('float')/255

        # inputs = [moving_images, fixed_images]
        # outputs = [fixed_images, zero_phi]

        yield [fixed_images, moving_images], fixed_images


def split_resize_data():
    from sklearn.model_selection import train_test_split
    from shutil import copyfile
    IMG_SIZE = config.IMG_SIZE
    n = os.listdir(config.DATA_PATH_NPY)
    train, test = train_test_split(n, train_size=0.9)
    for file in train:
        img = np.load(os.path.join(config.DATA_PATH_NPY, file))
        if(img.shape[1] != IMG_SIZE):
            img = resize_img(img)
        np.save(os.path.join(config.TRAIN_PATH, file), img)
    for file in test:
        img = np.load(os.path.join(config.DATA_PATH_NPY, file))
        if(img.shape[1] != IMG_SIZE):
            img = resize_img(img)
        np.save(os.path.join(config.TEST_PATH, file), img)


def resize_img(img):
    IMG_SIZE = config.IMG_SIZE
    img_p = np.ndarray((39, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(39):
        img_p[i] = cv2.resize(img[i], (IMG_SIZE, IMG_SIZE))

    return img_p



def main():
    from sklearn.model_selection import train_test_split
    dir = os.listdir(config.TRAIN_PATH)
    x_train, x_val = train_test_split(dir, train_size=0.9)
    train_generator = data_generator(x_train)
    val_generator = data_generator(x_val)
    #split_resize_data()
    model = simple_cnn()
    if(config.USING_PRETRAINED_WEIGHTS):
        model = simple_cnn(pretrained_weights = config.WEIGHTS_PATH)

    opt = Adam(lr=3E-4)
    model.compile(loss=ncc_loss,
                optimizer=opt,
                metrics=[mse_loss])
    checkpoint = ModelCheckpoint(config.WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=config.PATIENCE, mode='min')
    callbacks_list = [checkpoint, earlystopping]
    history=model.fit(train_generator, epochs=config.NB_EPOCHS,
                    steps_per_epoch=config.STEPS_PER_EPOCH,
                    callbacks=callbacks_list,
                    validation_data=val_generator,
                    validation_steps=config.VAL_STEPS,
                    verbose=1)
    with open(config.HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    main()
