# download keras VGG19

import tensorflow as tf

import settings

model = tf.keras.applications.VGG19(include_top=False, input_shape=settings.INPUT_SIZE + (3,))
model.save('vgg19')
# model = tf.keras.saving.load_model('vgg19')
#
# print(model.summary())
# print('totally ' + str(len(model.layers)) + ' layers.')
