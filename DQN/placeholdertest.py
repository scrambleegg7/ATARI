#
import tensorflow as tf
import numpy as np


#
#
# this test intends to update weight parameter(s) with assign method
# weight(s) is(are) picked up with get_collection method
#     TRAINABLE_VARIABLES
# weight is defined with variable_scope to have scope
# on each function.



x_p = tf.placeholder("float",[None,5])

def getV1():

    with tf.variable_scope("my_weights"):
        initial_value = tf.truncated_normal([5,5], stddev=0.1)
        W = tf.get_variable("W",initializer=initial_value)

        res = tf.matmul( x_p, W )

    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"my_weights")

    return var, res

def getV2():


    with tf.variable_scope("my_weights2"):
        initial_value = tf.truncated_normal([5,5], stddev=0.1)
        W = tf.get_variable("W2",initializer=initial_value)

        res = tf.matmul(x_p, W)

    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"my_weights2")

    return var, res

tfv1,res1 = getV1()
for v in tfv1:
    print("** tfv1 var name", v.name)

tfv2,res2 = getV2()
for v in tfv2:
    print("** tfv2 var name", v.name)

# copy tfv1 --> tfv2
copyTargetQNetworkOperation = [v.assign( tfv1[i] ) for i, v in enumerate(tfv2)]


with tf.Session() as sess:

    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
    sess.run(init_op)
    #saver = tf.train.Saver()

    x = np.ones((1,5))
    feed_dict = {x_p:x}

    sess_v1 = sess.run( tfv1 )
    for v1 in sess_v1:
        print(v1.shape)
        print(v1)

    sess_v2 = sess.run( tfv2 )
    for v2 in sess_v2:
        print(v2.shape)
        print(v2)

    sess_res1 = sess.run(res1, feed_dict=feed_dict)
    sess_res2 = sess.run(res2, feed_dict=feed_dict)
    print(sess_res1)
    print(sess_res2)

    sess.run(copyTargetQNetworkOperation)
    sess_res1 = sess.run(res1, feed_dict=feed_dict)
    sess_res2 = sess.run(res2, feed_dict=feed_dict)
    print(sess_res1)
    print(sess_res2)
