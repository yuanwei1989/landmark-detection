# Author: Yuanwei Li (29 Sep 2018)
#
# Multiple landmark detection in 3D ultrasound images of fetal head
# Simple network inference without evaluations.
#
# Reference
# Fast Multiple Landmark Localisation Using a Patch-based Iterative Network
# https://arxiv.org/abs/1806.06987
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import input_data, shape_model_func, patch, plane

np.random.seed(0)

class Config(object):
    """Inference configurations."""
    model_dir = './cnn_model'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    # Testing parameters
    landmark_count = 10     # Number of landmarks
    box_size = 101          # patch size (odd number)
    max_test_steps = 10     # Number of inference steps
    num_random_init = 5     # Number of random initialisations used


def main():
    config = Config()

    # Load shape model
    shape_model = shape_model_func.load_shape_model(config.shape_model_file, config.eigvec_per)

    # Load one test image
    img, pix_dim = input_data.extract_image('./data/Images/data1.nii.gz')

    # Load CNN model
    cnn_model = {}
    cnn_model['sess'] = tf.InteractiveSession()
    g = tf.get_default_graph()
    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(config.model_dir) + '.meta')
    saver.restore(cnn_model['sess'], tf.train.latest_checkpoint(config.model_dir))
    cnn_model['action_ind'] = g.get_collection('action_ind')[0]  # classification task: predict action indices
    cnn_model['yc'] = g.get_collection('yc')[0]                  # classification task: action probabilities (before softmax)
    cnn_model['yr'] = g.get_collection('yr')[0]                  # regression task: predict distance to gt landmark
    cnn_model['x'] = g.get_collection('x')[0]
    cnn_model['keep_prob'] = g.get_collection('keep_prob')[0]

    # Network inference
    landmarks, slice_tv, slice_tc, landmarks_proj_tv, landmarks_proj_tc = predict_landmarks(img[..., np.newaxis],
                                                                                            pix_dim,
                                                                                            config,
                                                                                            shape_model,
                                                                                            cnn_model)
    print("\nPredicted landmarks: \n{}".format(landmarks))
    cnn_model['sess'].close()

    # Plot results
    plt.ion()
    fig = plt.figure()
    plt.subplot(121)
    plt.title('{}'.format('TV plane'))
    plt.axis('off')
    plt.imshow(slice_tv, cmap='gray')
    marker_size = np.abs(landmarks_proj_tv[:, 2]) * 5
    plt.scatter(landmarks_proj_tv[:, 1], landmarks_proj_tv[:, 0], c=[0, 1, 0], s=marker_size, alpha=0.3)
    plt.scatter(landmarks_proj_tv[:, 1], landmarks_proj_tv[:, 0], c=[0, 1, 0], s=6)
    plt.subplot(122)
    plt.title('{}'.format('TC plane'))
    plt.axis('off')
    plt.imshow(slice_tc, cmap='gray')
    marker_size = np.abs(landmarks_proj_tc[:, 2]) * 5
    plt.scatter(landmarks_proj_tc[:, 1], landmarks_proj_tc[:, 0], c=landmarks_proj_tc.shape[0]*[[0, 1, 0]], s=marker_size, alpha=0.3)
    plt.scatter(landmarks_proj_tc[:, 1], landmarks_proj_tc[:, 0], c=landmarks_proj_tc.shape[0]*[[0, 1, 0]], s=6)
    raw_input()
    plt.close(fig)


def predict_landmarks(image, pix_dim, config, shape_model, cnn_model):
    """Predict landmarks.

    Args:
      image: image with dimensions=[width, height, depth, channel].
      pix_dim: pixel spacing
      config: test parameters
      shape_model: structure containing shape model
      cnn_model: dictionary of tensorflow nodes required for inference

    Returns:
      landmarks: Predicted landmark coordinates in voxel. [num_landmarks, 3] where num_landmarks=10
      slice_tv: image of the transventricular plane fitted to the landmarks
      slice_tc: image of the transcerebellar plane fitted to the landmarks
      landmarks_proj_tv: landmarks coordinates on slice_tv. [num_landmarks, 3] where num_landmarks=8
      landmarks_pro_tc: landmarks coordinates on slice_tc. [num_landmarks, 3] where num_landmarks=3

    """
    num_landmarks = config.landmark_count
    max_test_steps = config.max_test_steps
    box_size = config.box_size
    box_r = int((box_size - 1) / 2)
    num_examples = config.num_random_init + 1

    # Initialise shape parameters. Initialise b=0 and 5 random initialisations.
    num_shape_params = shape_model['Evectors'].shape[1]
    b = np.zeros((num_examples, num_shape_params))
    bounds = config.sd * np.sqrt(shape_model['Evalues'])
    b[1:num_examples] = np.random.rand(config.num_random_init, num_shape_params) * 2 * bounds - bounds

    # convert b to landmarks
    landmarks = shape_model_func.b2landmarks(b, shape_model)  # landmarks=[num_examples, num_landmarks, 3]

    # Extract patches from landmarks
    patches = np.zeros((num_examples, box_size, box_size, 3 * num_landmarks))
    for j in xrange(num_examples):
        patches[j] = patch.extract_patch_all_landmarks(image, landmarks[j], box_r)  # patches=[num_examples, box_size, box_size, 3*num_landmarks]

    landmarks_all_steps = np.zeros((max_test_steps + 1, num_examples, num_landmarks, 3))
    landmarks_all_steps[0] = landmarks

    for j in xrange(max_test_steps):  # find path of landmark iteratively
        # Predict CNN outputs
        action_ind_val, yc_val, yr_val = cnn_model['sess'].run([cnn_model['action_ind'],
                                                                cnn_model['yc'],
                                                                cnn_model['yr']],
                                                               feed_dict={cnn_model['x']: patches,
                                                                          cnn_model['keep_prob']: 1.0})

        # Compute classification probabilities
        action_prob = np.exp(yc_val - np.expand_dims(np.amax(yc_val, axis=1), 1))
        action_prob = action_prob / np.expand_dims(np.sum(action_prob, axis=1), 1)  # action_prob=[num_examples, 2*num_shape_params]

        # Update b values by multiplying classification probabilities with regressed distances.
        b = b - yr_val * np.amax(np.reshape(action_prob, (b.shape[0], b.shape[1], 2)), axis=2)

        # Convert b to landmarks
        landmarks = shape_model_func.b2landmarks(b, shape_model)  # landmarks=[num_examples, num_landmarks, 3]
        landmarks_all_steps[j + 1] = landmarks

        # Extract patches from landmarks
        for k in xrange(num_examples):
            patches[k] = patch.extract_patch_all_landmarks(image, landmarks[k], box_r)  # patches=[num_examples, box_size, box_size, 3*num_landmarks]

    # Compute mean of all initialisations as final landmark prediction
    landmarks = np.mean(landmarks_all_steps[-1, :, :, :], axis=0)  # landmarks_mean=[num_landmarks, 3]

    # Convert the scaling back to that of the original image.
    landmarks = landmarks * pix_dim / 0.5
    image = scipy.ndimage.zoom(image[:, :, :, 0], pix_dim / 0.5)

    # Extract TV and TC planes from landmarks
    landmarks_tv = np.vstack((landmarks[0:7], landmarks[9]))
    landmarks_tc = landmarks[7:10]
    slice_tv, landmarks_proj_tv = extract_plane(image, landmarks_tv, 'tv')
    slice_tc, landmarks_proj_tc = extract_plane(image, landmarks_tc, 'tc')

    return landmarks, slice_tv, slice_tc, landmarks_proj_tv, landmarks_proj_tc


def extract_plane(image, landmarks, plane_name):
    """Extract 2D plane image and project landmarks onto that plane.

    Args:
      image: 3D volume. [width, height, depth]
      landmarks: predicted landmarks. [num_landmarks, 3]
      plane_name: 'tv' or 'tc'

    Returns:
      slice: extracted plane image
      landmarks_proj: projected landmarks [num_landmarks, 3].
                      First and second columns give the coordinates on the 2D plane
                      Third column gives the distance from the 2D plane.

    """
    if plane_name == 'tv':
        assert (landmarks.shape[0] == 8)
    elif plane_name == 'tc':
        assert (landmarks.shape[0] == 3)

    img_siz = np.array(image.shape)
    img_c = (img_siz - 1) / 2.0

    # Extract plane image
    mat = plane.extract_tform(landmarks, plane_name)
    mat[:3, 3] = mat[:3, 3] - img_c
    slice, _ = plane.extract_plane_from_pose(image, mat, img_siz[:2], 1)

    # Project landmarks
    mat_inv = np.linalg.inv(mat)
    landmarks_proj = landmarks - img_c
    landmarks_proj = np.matmul(mat_inv[:3, :3], landmarks_proj.transpose()).transpose() + mat_inv[:3, 3]
    landmarks_proj[:, :2] = landmarks_proj[:, :2] + img_c[:2]

    return slice, landmarks_proj


if __name__ == '__main__':
    main()
