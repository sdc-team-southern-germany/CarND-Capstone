# Copyright (c) 2017 Andrey Voroshilov (modified)

#!/usr/bin/python
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import scipy.io
import time
import helper
import cv2

from PIL import Image
from argparse import ArgumentParser
from tensorflow.contrib.layers import flatten

# Get the log directory
LOG_DIR = os.getcwd()+ "/logs"

sigma = 0.1
mu = 0


def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32

# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)

    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))

    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name, load_vars=True, weights=None, biases=None):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    if load_vars:
        w, b = get_weights_biases(preloaded, layer_name)
        #if cluster_name == 'fire9':
            #print('~~~~~~~~~~~~~~:', layer_name)
            #print('weights', cluster_name, w.shape)
            #print('biases', cluster_name, b.shape)
    else:
        w = weights['squeeze1x1']
        b = biases['squeeze1x1']
    x = _conv_layer(net, layer_name + '_conv', x, w, b, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)

    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    if load_vars:
        w, b = get_weights_biases(preloaded, layer_name)
        #if cluster_name == 'fire9':
            #print('~~~~~~~~~~~~~~:', layer_name)
            #print('weights', cluster_name, w.shape)
            #print('biases', cluster_name, b.shape)
    else:
        w = weights['expand1x1']
        b = biases['expand1x1']
    x_l = _conv_layer(net, layer_name + '_conv', x, w, b, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    if load_vars:
        w, b = get_weights_biases(preloaded, layer_name)
        #if cluster_name == 'fire9':
            #print('~~~~~~~~~~~~~~:', layer_name)
            #print('weights', cluster_name, w.shape)
            #print('biases', cluster_name, b.shape)
    else:
        w = weights['expand3x3']
        b = biases['expand3x3']
    x_r = _conv_layer(net, layer_name + '_conv', x, w, b, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)

    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x

    return x

def net_preloaded(preloaded, input_image, pooling, keep_prob=None):
    net = {}
    cr_time = time.time()

#    x = tf.cast(input_image, get_dtype_tf())
    x = input_image

    # Feature extractor
    #####################
    with tf.name_scope("FeatureExtractor"):
        # conv1 cluster
        layer_name = 'conv1'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
        x = _act_layer(net, layer_name + '_actv', x)
        x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

        # fire2 + fire3 clusters
        x = fire_cluster(net, x, preloaded, cluster_name='fire2', load_vars=True)
        fire2_bypass = x
        x = fire_cluster(net, x, preloaded, cluster_name='fire3', load_vars=True)
        x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

        # fire4 + fire5 clusters
        x = fire_cluster(net, x, preloaded, cluster_name='fire4', load_vars=True)
        fire4_bypass = x
        x = fire_cluster(net, x, preloaded, cluster_name='fire5', load_vars=True)
        x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

        # remainder (no pooling)
        x = fire_cluster(net, x, preloaded, cluster_name='fire6', load_vars=True)
        fire6_bypass = x
        
        # fire7 cluster
        weights_fire7_squeeze1x1 = np.random.normal(mu, sigma, size=(1, 1, 384, 48))
        weights_fire7_expand1x1  = np.random.normal(mu, sigma, size=(1, 1, 48, 192))
        weights_fire7_expand3x3  = np.random.normal(mu, sigma, size=(1, 1, 48, 192))
        biases_fire7_squeeze1x1  = np.random.normal(mu, sigma, size=(48))
        biases_fire7_expand1x1   = np.random.normal(mu, sigma, size=(192))
        biases_fire7_expand3x3   = np.random.normal(mu, sigma, size=(192))
        weights_fire7 = {'squeeze1x1': weights_fire7_squeeze1x1, 'expand1x1': weights_fire7_expand1x1, 'expand3x3': weights_fire7_expand3x3}
        biases_fire7  = {'squeeze1x1': biases_fire7_squeeze1x1, 'expand1x1': biases_fire7_expand1x1, 'expand3x3': biases_fire7_expand3x3}
        x = fire_cluster(net, x, preloaded, cluster_name='fire7', load_vars=True, weights=weights_fire7, biases=biases_fire7)
        
        # fire8 cluster
        weights_fire8_squeeze1x1 = np.random.normal(mu, sigma, size=(1, 1, 384, 64))
        weights_fire8_expand1x1  = np.random.normal(mu, sigma, size=(1, 1, 64, 256))
        weights_fire8_expand3x3  = np.random.normal(mu, sigma, size=(1, 1, 64, 256))
        biases_fire8_squeeze1x1  = np.random.normal(mu, sigma, size=(64))
        biases_fire8_expand1x1   = np.random.normal(mu, sigma, size=(256))
        biases_fire8_expand3x3   = np.random.normal(mu, sigma, size=(256))
        weights_fire8 = {'squeeze1x1': weights_fire8_squeeze1x1, 'expand1x1': weights_fire8_expand1x1, 'expand3x3': weights_fire8_expand3x3}
        biases_fire8  = {'squeeze1x1': biases_fire8_squeeze1x1, 'expand1x1': biases_fire8_expand1x1, 'expand3x3': biases_fire8_expand3x3}
        x = fire_cluster(net, x, preloaded, cluster_name='fire8', load_vars=True, weights=weights_fire8, biases=biases_fire8)

        # fire9 cluster
        weights_fire9_squeeze1x1 = np.random.normal(mu, sigma, size=(1, 1, 512, 64))
        weights_fire9_expand1x1  = np.random.normal(mu, sigma, size=(1, 1, 64, 256))
        weights_fire9_expand3x3  = np.random.normal(mu, sigma, size=(1, 1, 64, 256))
        biases_fire9_squeeze1x1  = np.random.normal(mu, sigma, size=(64))
        biases_fire9_expand1x1   = np.random.normal(mu, sigma, size=(256))
        biases_fire9_expand3x3   = np.random.normal(mu, sigma, size=(256))
        weights_fire9 = {'squeeze1x1': weights_fire9_squeeze1x1, 'expand1x1': weights_fire9_expand1x1, 'expand3x3': weights_fire9_expand3x3}
        biases_fire9  = {'squeeze1x1': biases_fire9_squeeze1x1, 'expand1x1': biases_fire9_expand1x1, 'expand3x3': biases_fire9_expand3x3}
        x = fire_cluster(net, x, preloaded, cluster_name='fire9', load_vars=True, weights=weights_fire9, biases=biases_fire9)

    # Classifier
    #####################
    '''
    Traffic Light Classifier classes
    0 = RED
    1 = YELLOW
    2 = GREEN
    4 = UNKNOWN
    '''
    with tf.name_scope("Classifier"):
        # Dropout [use value of 50% when training]
        x = tf.nn.dropout(x, keep_prob)

        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        #weights, biases = get_weights_biases(preloaded, layer_name)
        weights = tf.Variable(tf.truncated_normal(shape=(1, 1, 512, 512), mean = mu, stddev = sigma))
        biases  = tf.Variable(tf.zeros(512))
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)

        # Global Average Pooling
        x = _pool_layer(net, 'classifier_pool', x, 'avg', size=(13, 13), stride=(1, 1), padding='VALID')

        # Flatten. Input = 1x1x1x1000. Output = 1x1000.
        fc0 = flatten(x)

        # Fully Connected. Input = 1000. Output = 100.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(512, 100), mean = 0, stddev = 0.1))
        fc1_b = tf.Variable(tf.zeros(100))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        # Activation.
        fc1 = tf.nn.relu(fc1)

        # Fully Connected. Input = 100. Output = 20.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 20), mean = 0, stddev = 0.1))
        fc2_b = tf.Variable(tf.zeros(20))
        fc2   = tf.matmul(fc1, fc2_W) + fc2_b
        # Activation.
        fc2 = tf.nn.relu(fc2)

        # Fully Connected. Input = 20. Output = 4.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(20, 4), mean = 0, stddev = 0.1))
        fc3_b = tf.Variable(tf.zeros(4))
        fc3   = tf.matmul(fc2, fc3_W) + fc3_b
        # Activation.
        logits = tf.nn.relu(fc3)

        net['classifier_actv'] = logits

    print("Network instance created: %fs" % (time.time() - cr_time))

    return net, logits

def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input, weights, strides=(1, stride[0], stride[1], 1),
                padding=padding)
        x = tf.nn.bias_add(conv, bias)
        net[name] = x
        return x

def _act_layer(net, name, input):
    with tf.name_scope(name):
        x = tf.nn.relu(input)
        net[name] = x
        return x

def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    with tf.name_scope(name):
        if pooling == 'avg':
            x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                    padding=padding)
        else:
            x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                    padding=padding)
        net[name] = x
        return x

def main():
    print

    # Loading network
    data, sqz_mean = load_net('./SqueezeNet/sqz_full.mat')

    # Hyperparameters
    epochs = 10
    lr = 1e-4
    kp = 0.5

    # Load training data generator
    data_dir_LARA = os.path.join(os.getcwd(),"data")
    get_batches_fn, X_test, y_test = helper.gen_batch_function_LARA(data_dir_LARA)

    # Placeholders
    images        = tf.placeholder(dtype=tf.float32, shape=(None, helper.HEIGHT, helper.WIDTH, 3), name="input_images")
    labels        = tf.placeholder(dtype=tf.int32, shape=None, name="labels")
    keep_prob     = tf.placeholder(dtype=tf.float32, name="keep_prob")
    learning_rate = tf.placeholder(dtype=tf.float32, name="lrate")

    # SqueezeNet model
    model, logits = net_preloaded(data, images, 'max', keep_prob)

    # Loss and Training operations
    with tf.name_scope("Retraining"):
        cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    # Accuracy operation
    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy_operation)

    # Evaluate the loss and accuracy of the model
    def evaluate(X_data, y_data, batch_size, sess):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]

            accuracy = sess.run(accuracy_operation, feed_dict={images: batch_x,
                                                               labels: batch_y,
                                                               keep_prob: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    # Save variables
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    # Simple classification
    print
    with tf.Session(config=config) as sess:
        print

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        summ = tf.summary.merge_all()
        # Tensorflow visualization
        summ_writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)

        print('Training...')
        for epoch in range(epochs):
            gen = get_batches_fn()
            for X_train, y_train in gen:

                _, loss = sess.run([training_operation, cross_entropy_loss],
                                   feed_dict={images: X_train,
                                              labels: y_train,
                                              keep_prob: kp,
                                              learning_rate: lr}
                                  )
            print('Epoch {}: loss = {}'.format(epoch+1, loss))

        # Test accuracy
        test_accuracy = evaluate(X_test, y_test, helper.batch_size, sess)
        print("Test Accuracy = {:.2f}%".format(test_accuracy*100))

        # Save the variables to disk.
        saver.save(sess, "model/model")
        print("Model saved.")

if __name__ == '__main__':
    main()
