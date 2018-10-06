"""Functions for calculations with shape model."""

import numpy as np
import scipy.io as sio


def load_shape_model(shape_model_file, eigvec_per):
    """Load the shape model.

    Args:
    shape_model_file: path and file name of shape model file (.mat)
    eigvec_per: Percentage of eigenvectors to keep (0-1)

    Returns:
    shape_model: a structure containing the shape model

    """
    mat_contents = sio.loadmat(shape_model_file)
    shape_model = mat_contents['ShapeData']
    shape_model = shape_model[0, 0]
    if (eigvec_per != 1):
        ind = np.nonzero(np.cumsum(shape_model['Evalues']) > np.sum(shape_model['Evalues']) * eigvec_per)[0][0]
        shape_model['Evectors'] = shape_model['Evectors'][:, :ind + 1]
        shape_model['Evalues'] = shape_model['Evalues'][:ind + 1]
    shape_model['Evalues'] = np.squeeze(shape_model['Evalues'])
    shape_model['x_mean'] = np.squeeze(shape_model['x_mean'])
    return shape_model


def landmarks2b(landmarks, shape_model):
    """Transform from landmarks to shape parameters using shape model.

    Args:
    landmarks: Landmark coordinates. [num_examples, num_landmarks, 3]
    shape_model: structure containing shape model

    Returns:
    b: shape model parameters. [num_examples, num_shape_params]

    """
    landmarks = np.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1]*landmarks.shape[2]))  # Reshape to [num_examples, 3*num_landmarks]
    b = np.transpose(np.matmul(np.transpose(shape_model['Evectors']), np.transpose(landmarks - shape_model['x_mean'])))
    return b


def b2landmarks(b, shape_model):
    """Transform from shape parameters to landmarks using shape model.

    Args:
    b: shape model parameters. [num_examples, num_shape_params]
    shape_model: structure containing shape model

    Returns:
    landmarks: Landmark coordinates. [num_examples, num_landmarks, 3]

    """
    landmarks = np.transpose(np.matmul(shape_model['Evectors'], np.transpose(b))) + shape_model['x_mean']
    landmarks = np.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1]/3, 3))
    return landmarks


def init_shape_params(num_random_init, k_top_b, sd, shape_model):
    """Initialise the shape parameters. Initialise b=0, and either:
    b = +/-(sd*sqrt(eigvalues)) for k_top_b principal components
    or
    b = num_random_init random initialisations

    Args:
      num_random_init: Number of random initialisations. If set to None, use fixed initialisation defined by k_top_b.
      k_top_b: The top k principal components to use for fixed initialisation. Only valid if num_random_init is set to None.
      sd: standard deviation away from eigenvalues
      shape_model: needed for the eigenvalues

    Returns:
      b: initialisation of b. [num_examples, num_shape_params]

    """
    num_shape_params = shape_model['Evectors'].shape[1]

    if num_random_init is None:     # Using fixed initialisations
        # No deviation
        b = np.zeros((2*k_top_b+1, num_shape_params))
        # Positive deviation
        b[np.arange(1, 1 + k_top_b), np.arange(k_top_b)] = sd * np.sqrt(shape_model['Evalues'][:k_top_b])
        # Negative deviation
        b[np.arange(1+k_top_b, 1+2*k_top_b), np.arange(k_top_b)] = -sd * np.sqrt(shape_model['Evalues'][:k_top_b])
    else:                           # Using random initialisations
        b = np.zeros((num_random_init+1, num_shape_params))
        bounds = sd * np.sqrt(shape_model['Evalues'])
        b[1:num_random_init+1] = np.random.rand(num_random_init, num_shape_params) * 2 * bounds - bounds

    return b
