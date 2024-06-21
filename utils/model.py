from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
import tensorflow as tf

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    nima = Model(base_model.input, x)
    nima.load_weights('./weights/mobilenet_weights.h5')