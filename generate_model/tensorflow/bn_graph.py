import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.training.moving_averages import assign_moving_average

PB_FILE_PATH = './model.pb'

bn_input = tf.placeholder(tf.float32, shape=(1,64,64,3), name='input')

fused_bn_output = tf.layers.batch_normalization(
        bn_input,
        beta_initializer=tf.random_uniform_initializer(),
        gamma_initializer=tf.random_uniform_initializer(),
        moving_mean_initializer=tf.random_uniform_initializer(),
        moving_variance_initializer=tf.random_uniform_initializer(),
        name='tf.layers.batch_normlization')


params_shape = bn_input.shape[-1:]
moving_mean = tf.get_variable(name='mean', shape=params_shape,
                              initializer=tf.random_uniform_initializer)
moving_variance = tf.get_variable('variance', params_shape,
                                  initializer=tf.random_uniform_initializer)
offset = tf.get_variable('offset', params_shape,
                         initializer=tf.random_uniform_initializer)
scale = tf.get_variable('scale', params_shape,
                        initializer=tf.random_uniform_initializer)
nn_bn_output = tf.nn.batch_normalization(
        bn_input,
        mean=moving_mean,
        variance=moving_variance,
        offset=offset,
        scale=scale,
        variance_epsilon=0.01,
        name='tf.nn.batch_normalization')

add_output = tf.add(fused_bn_output, nn_bn_output, name='add_output')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add_output'])
    with tf.gfile.FastGFile(PB_FILE_PATH, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    random_in = np.random.rand(1,64,64,3)
    print(sess.run(fused_bn_output, feed_dict={bn_input: random_in}))
    print(sess.run(nn_bn_output, feed_dict={bn_input: random_in}))
    print(sess.run(add_output, feed_dict={bn_input: random_in}))
