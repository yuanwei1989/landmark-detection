"""Functions for visualisations."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plane


def project_landmarks(image, landmarks, landmarks_gt, plane_name):
    """Extract 2D plane image and project landmarks onto that plane.

    Args:
      image: 3D volume. [width, height, depth]
      landmarks: predicted landmarks. [num_landmarks, 3]
      landmarks_gt: GT landmarks [num_landmarks, 3]
      plane_name: 'tv' or 'tc'

    Returns:
      slice: extracted plane image
      landmarks_proj: projected landmarks [num_landmarks, 3]
      landmarks_gt_proj: projected GT landmarks [num_landmarks, 3]

    """
    if plane_name == 'tv':
        assert(landmarks_gt.shape[0] == 8)
    elif plane_name == 'tc':
        assert(landmarks_gt.shape[0] == 3)
    img_siz = np.array(image.shape)
    img_c = (img_siz - 1) / 2.0

    # Extract plane image
    mat = plane.extract_tform(landmarks_gt, plane_name)
    mat[:3, 3] = mat[:3, 3] - img_c
    slice, _ = plane.extract_plane_from_pose(image, mat, img_siz[:2], 1)

    # Project landmarks
    mat_inv = np.linalg.inv(mat)
    landmarks_gt_proj = landmarks_gt - img_c
    landmarks_gt_proj = np.matmul(mat_inv[:3, :3], landmarks_gt_proj.transpose()).transpose() + mat_inv[:3, 3]
    landmarks_gt_proj[:, :2] = landmarks_gt_proj[:, :2] + img_c[:2]
    landmarks_proj = landmarks - img_c
    landmarks_proj = np.matmul(mat_inv[:3, :3], landmarks_proj.transpose()).transpose() + mat_inv[:3, 3]
    landmarks_proj[:, :2] = landmarks_proj[:, :2] + img_c[:2]

    return slice, landmarks_proj, landmarks_gt_proj


def plot_landmarks_2d(save_dir, train, name, image, landmarks, landmarks_gt):
    """Plot predicted landmarks overlaid on fitted planes.

    Args:
      save_dir: Directory storing the results.
      train: train or test dataset
      name: name of patient.
      image: 3D volume. [width, height, depth]
      landmarks: predicted landmarks. [num_landmarks, 3], num_landmarks=10
      landmarks_gt: GT landmarks [num_landmarks, 3], num_landmarks=10

    """
    landmarks_gt_tv = np.vstack((landmarks_gt[0:7], landmarks_gt[9]))
    landmarks_gt_tc = landmarks_gt[7:10]
    landmarks_tv = np.vstack((landmarks[0:7], landmarks[9]))
    landmarks_tc = landmarks[7:10]

    slice_tv, landmarks_tv_proj, landmarks_gt_tv_proj = project_landmarks(image, landmarks_tv, landmarks_gt_tv, 'tv')
    slice_tc, landmarks_tc_proj, landmarks_gt_tc_proj = project_landmarks(image, landmarks_tc, landmarks_gt_tc, 'tc')

    fig = plt.figure()
    plt.subplot(121)
    plt.title('{}'.format('TV plane'))
    plt.axis('off')
    plt.imshow(slice_tv, cmap='gray')
    marker_size = np.abs(landmarks_tv_proj[:, 2]) * 5
    plt.scatter(landmarks_tv_proj[:, 1], landmarks_tv_proj[:, 0], c=[0, 1, 0], s=marker_size, alpha=0.3)
    plt.scatter(landmarks_tv_proj[:, 1], landmarks_tv_proj[:, 0], c=[0, 1, 0], s=6)
    plt.scatter(landmarks_gt_tv_proj[:, 1], landmarks_gt_tv_proj[:, 0], c=[1, 0, 0], s=6)
    plt.subplot(122)
    plt.title('{}'.format('TC plane'))
    plt.axis('off')
    plt.imshow(slice_tc, cmap='gray')
    marker_size = np.abs(landmarks_tc_proj[:, 2]) * 5
    plt.scatter(landmarks_tc_proj[:, 1], landmarks_tc_proj[:, 0], c=landmarks_tc_proj.shape[0]*[[0, 1, 0]], s=marker_size, alpha=0.3)
    plt.scatter(landmarks_tc_proj[:, 1], landmarks_tc_proj[:, 0], c=landmarks_tc_proj.shape[0]*[[0, 1, 0]], s=6)
    plt.scatter(landmarks_gt_tc_proj[:, 1], landmarks_gt_tc_proj[:, 0], c=landmarks_tc_proj.shape[0]*[[1, 0, 0]], s=6)
    if train:
        save_dir = os.path.join(save_dir, 'train')
    else:
        save_dir = os.path.join(save_dir, 'test')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, name+'.png'), bbox_inches='tight')
    plt.close(fig)


def plot_landmarks_3d(save_dir, train, name, landmarks_mean, landmarks_gt, dim):
    """Plot predicted landmarks in 3D space

    Args:
      save_dir: Directory storing the results.
      train: train or test dataset
      name: name of patient.
      landmarks_mean: predicted landmarks. [num_landmarks, 3]
      landmarks_gt: GT landmarks [num_landmarks, 3]
      dim: volume size [3]

    """
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(landmarks_mean[:, 0], landmarks_mean[:, 1], landmarks_mean[:, 2], 'g.')
    ax.plot(landmarks_gt[:, 0], landmarks_gt[:, 1], landmarks_gt[:, 2], 'r.')
    ax.set_title('{}'.format(name))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, dim[0]])
    ax.set_ylim([0, dim[1]])
    ax.set_zlim([0, dim[2]])
    if train:
        save_dir = os.path.join(save_dir, 'train')
    else:
        save_dir = os.path.join(save_dir, 'test')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, name+'.png'), bbox_inches='tight')
    plt.close(fig)


def plot_landmarks_path(save_dir, train, name, landmarks_all_steps, landmarks_gt, dim):
    """Save predicted landmark paths as gif.

    Args:
      save_dir: Directory storing the results.
      train: train or test dataset
      name: name of patient.
      landmarks_all_steps: predicted landmarks. [max_test_steps + 1, num_examples, num_landmarks, 3]
      landmarks_gt: GT landmarks [num_landmarks, 3]
      dim: volume size [3]

    """
    c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
         'tab:olive', 'tab:cyan']
    num_landmarks = landmarks_all_steps.shape[2]
    max_test_steps = landmarks_all_steps.shape[0] - 1

    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_tight_layout(True)
    ax.plot(landmarks_gt[:, 0], landmarks_gt[:, 1], landmarks_gt[:, 2], 'rx')
    pt = []
    for j in xrange(num_landmarks):
        pt.append(ax.plot(landmarks_all_steps[0, :, j, 0], landmarks_all_steps[0, :, j, 1],
                           landmarks_all_steps[0, :, j, 2], '.', c=c[j])[0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, dim[0]])
    ax.set_ylim([0, dim[1]])
    ax.set_zlim([0, dim[2]])
    ax.set_title('{}'.format(name))

    def update(n):
        for j in xrange(num_landmarks):
            pt[j].set_data(landmarks_all_steps[n, :, j, 0], landmarks_all_steps[n, :, j, 1])  # x and y axis
            pt[j].set_3d_properties(zs=landmarks_all_steps[n, :, j, 2])  # z axis
        return pt

    anim = FuncAnimation(fig, update,
                         frames=np.arange(0, max_test_steps + 1, 1),
                         interval=400,
                         repeat_delay=3000,
                         repeat=True)
    if train:
        save_dir = os.path.join(save_dir, 'train')
    else:
        save_dir = os.path.join(save_dir, 'test')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    anim.save(os.path.join(save_dir, name + '.gif'), dpi=80, writer='imagemagick')
    plt.close(fig)
