# Author: Yuanwei Li (29 Sep 2018)
#
# Multiple landmark detection in 3D ultrasound images of fetal head
# Network training
#
# Reference
# Fast Multiple Landmark Localisation Using a Patch-based Iterative Network
# https://arxiv.org/abs/1806.06987
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from utils import input_data, shape_model_func, network, patch


class Config(object):
    """Training configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    log_dir = './logs'
    model_dir = './cnn_model'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = 10     # Number of landmarks
    landmark_unwant = [0, 8, 9, 13, 14, 15]     # list of unwanted landmark indices
    # Training parameters
    resume = False          # Whether to train from scratch or resume previous training
    box_size = 101          # patch size (odd number)
    alpha = 0.5             # Weighting given to the loss (0<=alpha<=1). loss = alpha*loss_c + (1-alpha)*loss_r
    learning_rate = 0.001
    max_steps = 100000      # Number of steps to train
    save_interval = 25000   # Number of steps in between saving each model
    batch_size = 64         # Training batch size
    dropout = 0.5


def main():
    config = Config()

    # Load shape model
    shape_model = shape_model_func.load_shape_model(config.shape_model_file, config.eigvec_per)
    num_cnn_output_c = 2 * shape_model['Evectors'].shape[1]
    num_cnn_output_r = shape_model['Evectors'].shape[1]

    # Load images and landmarks
    data = input_data.read_data_sets(config.data_dir,
                                     config.label_dir,
                                     config.train_list_file,
                                     config.test_list_file,
                                     config.landmark_count,
                                     config.landmark_unwant,
                                     shape_model)

    # Build graph
    print("Building graph...")
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, config.box_size, config.box_size, 3*config.landmark_count], name='x-input')
        tf.add_to_collection('x', x)
        yc_ = tf.placeholder(tf.float32, [None, num_cnn_output_c], name='yc-input')     # one hot vector for classification labels (positive or negative for each shape parameter)
        tf.add_to_collection('yc_', yc_)
        yr_ = tf.placeholder(tf.float32, [None, num_cnn_output_r], name='yr-input')     # regression output
        tf.add_to_collection('yr_', yr_)

    # Define CNN model
    yc, yr, keep_prob = network.cnn(x, num_cnn_output_c, num_cnn_output_r)
    tf.add_to_collection('yc', yc)
    tf.add_to_collection('yr', yr)
    tf.add_to_collection('keep_prob', keep_prob)

    # Define prediction
    with tf.name_scope('prediction'):
        action_ind = tf.argmax(yc, 1)
        tf.add_to_collection('action_ind', action_ind)
        action_prob = tf.nn.softmax(yc)
        tf.add_to_collection('action_prob', action_prob)

    # Define loss
    with tf.name_scope('loss'):
        # Loss weight, alpha
        alpha = tf.placeholder(tf.float32, name='alpha')
        tf.add_to_collection('alpha', alpha)

        # Classification loss (cross entropy)
        loss_c = alpha * tf.nn.softmax_cross_entropy_with_logits(labels=yc_, logits=yc)
        loss_c = tf.reduce_mean(loss_c)
        tf.add_to_collection('loss_c', loss_c)
        tf.summary.scalar('loss_c', loss_c)

        # Regresssion loss (MSE)
        loss_r = (1 - alpha) * tf.reduce_mean(tf.pow(yr - yr_, 2), axis=1)
        loss_r = tf.reduce_mean(loss_r)
        tf.add_to_collection('loss_r', loss_r)
        tf.summary.scalar('loss_r', loss_r)

        # Combined loss
        loss = loss_c + loss_r
        tf.add_to_collection('loss', loss)
        tf.summary.scalar('loss', loss)

    # Define optimizer
    with tf.name_scope('train'):
        # # learning rate decreases over time
        # global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = config.learning_rate
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 250, 0.5, staircase=True)
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # tf.summary.scalar('learning_rate', learning_rate)
        # Constant learning rate
        train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
        tf.add_to_collection('train_step', train_step)

    with tf.name_scope('performance'):
        # classification accuracy
        correct_classification = tf.equal(tf.argmax(yc, 1), tf.argmax(yc_, 1))
        correct_classification = tf.cast(correct_classification, tf.float32)
        accuracy = tf.reduce_mean(correct_classification)
        tf.add_to_collection('accuracy', accuracy)
        tf.summary.scalar('accuracy', accuracy)
        # Regression squared distance error is given by loss_r


    # Run training
    print("Start training...")
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(config.log_dir + '/test')

    if config.resume:
        # Resume previous training
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
        saver = tf.train.Saver(max_to_keep=20)
        ite_start = int(tf.train.latest_checkpoint(config.model_dir).split('-')[-1])
        ite_end = ite_start + config.max_steps
    else:
        # Start new training
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=20)
        ite_start = 0
        ite_end = config.max_steps

    for i in xrange(ite_start, ite_end):
        patches_train, actions_train, dbs_train, _ = get_train_pairs(config.batch_size,
                                                                     data.train.images,
                                                                     data.train.shape_params,
                                                                     config.box_size,
                                                                     num_cnn_output_c,
                                                                     num_cnn_output_r,
                                                                     shape_model,
                                                                     config.sd)

        if i % 1000 == 0:
            # Record summaries and test-set loss and accuracy
            patches_test, actions_test, dbs_test, _ = get_train_pairs(config.batch_size,
                                                                      data.test.images,
                                                                      data.test.shape_params,
                                                                      config.box_size,
                                                                      num_cnn_output_c,
                                                                      num_cnn_output_r,
                                                                      shape_model,
                                                                      config.sd)
            summary_test, l_test, lc_test, lr_test, acc_test = sess.run([merged, loss, loss_c, loss_r, accuracy], feed_dict={x: patches_test,
                                                                                                                             yc_: actions_test,
                                                                                                                             yr_: dbs_test,
                                                                                                                             keep_prob: 1.0,
                                                                                                                             alpha: config.alpha})
            test_writer.add_summary(summary_test, i)
            # Record summaries and train-set loss and accuracy
            summary_train, l_train, lc_train, lr_train, acc_train = sess.run([merged, loss, loss_c, loss_r, accuracy], feed_dict={x: patches_train,
                                                                                                                                  yc_: actions_train,
                                                                                                                                  yr_: dbs_train,
                                                                                                                                  keep_prob: 1.0,
                                                                                                                                  alpha: config.alpha})
            train_writer.add_summary(summary_train, i)
            print('Step {}: \ttrain: loss={:11.6f}, loss_c={:9.6f}, loss_r={:11.6f}, acc={:8.6f}. \ttest: loss={:11.6f}, loss_c={:9.6f}, loss_r={:11.6f}, acc={:8.6f}.'.format
                  (i, l_train, lc_train, lr_train, acc_train, l_test, lc_test, lr_test, acc_test))

        # Train one step
        _ = sess.run(train_step, feed_dict={x: patches_train,
                                            yc_: actions_train,
                                            yr_: dbs_train,
                                            keep_prob: config.dropout,
                                            alpha: config.alpha})

        # Save trained model
        if ((i+1) % config.save_interval) == 0:
            saver.save(sess, os.path.join(config.model_dir, 'model'), global_step=i+1)
            print("Trained model save successfully in {} at step {}".format(os.path.join(config.model_dir, 'model'), i+1))

    train_writer.close()
    test_writer.close()
    sess.close()


def get_train_pairs(batch_size, images, bs_gt, box_size, num_actions, num_regression_output, shape_model, sd):
    """Randomly sample image patches and corresponding ground truth classification and regression outputs.

    Args:
      batch_size: mini batch size
      images: list of img_count images. Each image is [width, height, depth, channel], [x,y,z,channel]
      bs_gt: Ground truth shape parameters. [img_count, num_shape_params]
      box_size: size of image patch. Scalar.
      num_actions: number of classification outputs
      num_regression_output: number of regression outputs
      shape_model: structure containing shape models
      sd: standard deviation of shape model. Bounds from which to sample bs.

    Returns:
      patches: 2D image patches, [batch_size, box_size, box_size, 3*num_landmarks]
      actions: Ground truth classification output. [batch_size, num_actions], each row is a one hot vector [positive or negative for each shape parameter]
      dbs: Ground truth regression output. [batch_size, num_regression_output]. dbs = bs - bs_gt.
      bs: sampled shape parameters [batch_size, num_regression_output]

    """
    img_count = len(images)
    num_landmarks = shape_model['Evectors'].shape[0] / 3
    box_r = int((box_size - 1) / 2)
    patches = np.zeros((batch_size, box_size, box_size, int(3*num_landmarks)), np.float32)
    actions_ind = np.zeros(batch_size, dtype=np.uint16)
    actions = np.zeros((batch_size, num_actions), np.float32)

    # get image indices randomly for a mini-batch
    ind = np.random.randint(img_count, size=batch_size)

    # Randomly sample shape parameters, bs
    bounds = sd * np.sqrt(shape_model['Evalues'])
    bs = np.random.rand(batch_size, num_regression_output) * 2 * bounds - bounds

    # Convert shape parameters to landmark
    landmarks = shape_model_func.b2landmarks(bs, shape_model)

    # Extract image patch
    for i in xrange(batch_size):
        image = images[ind[i]]
        patches[i] = patch.extract_patch_all_landmarks(image, landmarks[i], box_r)

    # Regression values (distances between predicted and GT shape parameters)
    dbs = bs - bs_gt[ind]

    # Extract classification labels as a one-hot vector
    max_db_ind = np.argmax(np.abs(dbs), axis=1)     # [batch_size]
    max_db = dbs[np.arange(dbs.shape[0]), max_db_ind]   # [batch_size]
    is_positive = (max_db > 0)
    actions_ind[is_positive] = max_db_ind[is_positive] * 2
    actions_ind[np.logical_not(is_positive)] = max_db_ind[np.logical_not(is_positive)] * 2 + 1
    actions[np.arange(batch_size), actions_ind] = 1

    return patches, actions, dbs, bs


if __name__ == '__main__':
    main()
