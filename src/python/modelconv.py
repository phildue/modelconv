from keras import Input, Model
from keras.layers import Conv2D
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util
# manually put back imported modules
import tempfile
import subprocess

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

K.set_learning_phase(0)
input = Input((416, 416, 3),name='Placeholder')
conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)
model = Model(input, conv1)
netin = [K.placeholder(name="Input", dtype=tf.float32, shape=(416,416,3))]
netout = [K.identity(model.outputs[0],"Prediction")]
sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Prediction"])

tflite_model = tf.contrib.lite.toco_convert(constant_graph, netin, netout)
open('model' + '.tflite', "wb").write(tflite_model)