import cv2
import keras.utils
import tensorflow as tf
import numpy as np

import settings
import pipe
import data

import os
import random
import math

import gc


def create_model(row_num, down_col_num, up_col_num):
    """Create a GridNet with the parameters"""
    def add_lateral_connection(filters, x):
        """Add right connection to x"""
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        return x

    def add_downsampling_connection(filters, x):
        """Add down connection to x"""
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        return x

    def add_upsampling_connection(filters, x):
        """Add up connection to x"""
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
        x = add_lateral_connection(filters, x)
        return x

    inputs = tf.keras.Input(settings.INPUT_SIZE + (3,))
    x00 = add_lateral_connection(32, inputs)

    # to record if a connection point has been already built, or a connection path will be duplicated
    grid = [[None] * (down_col_num + up_col_num) for i in range(row_num)]
    grid[0][0] = x00

    def add_up(x, y, filters):
        if x is None:
            return add_upsampling_connection(filters, y)
        if y is None:
            return add_lateral_connection(filters, x)
        return tf.keras.layers.Add()([add_lateral_connection(filters, x), add_upsampling_connection(filters, y)])

    def add_down(x, y, filters):
        if x is None:
            return add_downsampling_connection(filters, y)
        if y is None:
            return add_lateral_connection(filters, x)
        return tf.keras.layers.Add()([add_lateral_connection(filters, x), add_downsampling_connection(filters, y)])

    def get_grid(i, j):
        if i < 0 or i >= row_num or j < 0 or j >= down_col_num + up_col_num:
            return None
        if grid[i][j] is not None:
            return grid[i][j]
        if j < down_col_num:
            grid[i][j] = add_down(get_grid(i, j - 1), get_grid(i - 1, j), 32 * (i + 1))
            return grid[i][j]
        else:
            grid[i][j] = add_up(get_grid(i, j - 1), get_grid(i + 1, j), 32 * (i + 1))
            return grid[i][j]

    x = tf.keras.layers.PReLU()(get_grid(0, up_col_num + down_col_num - 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.PReLU()(x)
    outputs = tf.keras.layers.Conv2D(3, 3, padding="same")(x)

    return tf.keras.Model(inputs, outputs)


# init a new vgg19 model with custom output
# conv1 2, conv2 2, conv3 2, conv4 2, and conv5 2
def init_custom_vgg19():
    vgg = tf.keras.saving.load_model('vgg19')
    return tf.keras.Model(
        vgg.inputs,
        [
            vgg.layers[2].output,
            vgg.layers[5].output,
            vgg.layers[8].output,
            vgg.layers[13].output,
            vgg.layers[18].output,
        ]
    )


vgg19 = init_custom_vgg19()
vgg19_layer_weight_recip = [2.6, 4.8, 3.7, 5.6, 0.15]


def total_loss_func(y_true, y_pred):
    def pixel_loss_func(y_true, y_pred):
        return tf.norm(y_true - y_pred, ord=1)

    def feature_loss_func(y_true, y_pred):
        res = 0.0
        phi1, phi2 = vgg19(y_true), vgg19(y_pred)
        for i in range(len(vgg19_layer_weight_recip)):
            res += tf.norm(phi1[i] - phi2[i], ord=1) / vgg19_layer_weight_recip[i]
        return res

    return 0.01 * feature_loss_func(y_true, y_pred) + pixel_loss_func(y_true, y_pred)


def load_data(name_list_index_y) -> np.ndarray:
    """Load the n-th type data according to the name list in settings"""
    data_x = []
    data_y = []
    index_max = len(os.listdir('input')) // len(settings.name_list)
    for i in range(1, index_max + 1):
        name = str(i).zfill(3)
        path1 = 'input_face_cache/' + name + settings.name_list[0] + settings.file_format
        path2 = 'color_mask_cache/' + name + settings.name_list[name_list_index_y] + settings.file_format
        img_x, _, _ = data.standardize_face(cv2.imread(path1))
        img_y = data.prepare_mask(cv2.imread(path2))
        data_x.append(img_x)
        data_y.append(img_y)
    # data_x = np.asarray(data_x)
    # data_y = np.asarray(data_y)
    # return tf.image.per_image_standardization(data_x), tf.image.per_image_standardization(data_y)
    return data_x, data_y


def train(input_data_x, input_data_y, epoch=50):
    """Accepts 2 numpy arrays for x, y; In order!"""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
    )
    # pm = create_model(row_num=6, down_col_num=3, up_col_num=5)
    # pm.compile(optimizer=optimizer, loss=total_loss_func)

    pm = tf.keras.saving.load_model('model_III_b', custom_objects={'total_loss_func': total_loss_func})

    data_length = len(input_data_x)
    for e in range(epoch):
        index_list = list(range(data_length))
        random.shuffle(index_list)
        for i in range(data_length):
            print('EPOCH #' + str(e + 1) + ': ' + str(i + 1) + '/' + str(data_length))
            index = index_list[i]
            x = np.asarray([input_data_x[index]])
            y = np.asarray([input_data_y[index]])
            pm.fit(x, y)

    return pm


def evaluate(eval_path, model):
    """Predict all files in a folder"""
    eval_path += '/'
    try:
        os.mkdir(eval_path + 'result')
    except:
        pass
    for file in os.listdir(eval_path):
        if file == eval_path + 'result':
            continue
        try:
            face_ripper = data.FaceRipper(eval_path + file)
            face = face_ripper.face
            face, mean, dev = data.standardize_face(face)
            face = np.asarray([face])
            mask = model.predict(face)[0]
            mask = data.de_prepare_mask(mask)
            img = face_ripper.apply_mask_and_update_face(mask)
            # cv2.imwrite(eval_path + 'result/mask_' + file, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(eval_path + 'result/' + file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except:
            print('Error: ' + eval_path + file)


if __name__ == '__main__':
    # start a new model
    # pm = train(*load_data(3))
    # pm.save('model_III_b')
    # predict
    model = tf.keras.saving.load_model('model_III_b', custom_objects={'total_loss_func': total_loss_func})
    evaluate('final eval/test', model)
