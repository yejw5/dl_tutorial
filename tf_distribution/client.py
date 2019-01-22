import tensorflow as tf
import numpy as np

total_steps = 10000
train_data_size = 100

# placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")
# Creates weight
w = tf.Variable(0.0, name="weight")
# Creates biases
b = tf.Variable(0.0, name="reminder")
# init Variable
init_var = tf.global_variables_initializer()
# pred
with tf.device("/job:worker/task:0"):
    mul = tf.multiply(X, w)
with tf.device("/job:worker/task:1"):
    pred = mul -b
# loss function
loss = tf.square(Y - pred)
# global step
global_step = tf.train.get_or_create_global_step()
# Optimizer
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
        loss, var_list=[w, b], global_step=global_step)

# input parameter
train_X = np.linspace(-1, 1, train_data_size + 1)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10


worker_01 = "127.0.0.1:2222"

hooks=[tf.train.StopAtStepHook(last_step=total_steps)]
sess = tf.train.MonitoredTrainingSession(master="grpc://" + worker_01,
                                         is_chief=True,
                                         hooks=hooks)
sess.run(init_var)
index = 0
while not sess.should_stop():
    x = train_X[index % train_data_size]
    y = train_Y[index % train_data_size]
    sess.run([train_op, global_step], feed_dict={X: x, Y: y})
    index += 1
