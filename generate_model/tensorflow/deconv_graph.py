import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.training.moving_averages import assign_moving_average

PB_FILE_PATH = './deconv_model.pb'

deconv_input = tf.placeholder(tf.float32, shape=(1,64,64,3), name='input')
bias_input = tf.placeholder(tf.float32, shape=(5), name='input_bias')

deconv_output = tf.layers.conv2d_transpose(
        deconv_input,
        5, 3,
        use_bias=False,
        kernel_initializer=tf.random_uniform_initializer,
        bias_initializer=tf.random_uniform_initializer,
        name='tf.layers.conv2d_transpose')

bias_output = tf.nn.bias_add(deconv_output, bias_input, name='bias_output')

fused_bn_output = tf.layers.batch_normalization(
        bias_output,
        beta_initializer=tf.random_uniform_initializer(),
        gamma_initializer=tf.random_uniform_initializer(),
        moving_mean_initializer=tf.random_uniform_initializer(),
        moving_variance_initializer=tf.random_uniform_initializer(),
        name='tf.layers.batch_normlization')
mul_output = tf.multiply(fused_bn_output, tf.constant(2.0, name='scale'), name='mul_output')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['mul_output'])
    with tf.gfile.FastGFile(PB_FILE_PATH, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    random_in = np.random.rand(1,64,64,3)
    random_bias = np.random.rand(5)
    print(sess.run(mul_output, feed_dict={deconv_input: random_in, bias_input: random_bias}))
