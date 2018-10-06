"""Functions for writing results."""

import os
import numpy as np

def save_err(save_dir, train, names, dist_err, dist_err_mm):
    """save landmark distance errors in txt file.

    Args:
      save_dir: Directory storing the results.
      train: train or test dataset
      names: list of names of the patients.
      dist_err: distance error in pixel. [img_count, num_landmarks]
      dist_err_mm: distance error in mm. [img_count, num_landmarks]

    """
    img_count, num_landmarks = dist_err.shape
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if train:
        suffix = 'train'
    else:
        suffix = 'test'

    # Mean and std of distance error for each landmark
    dist_err_landmarks_mean = np.mean(dist_err, axis=0)
    dist_err_landmarks_std = np.std(dist_err, axis=0)
    dist_err_mm_landmarks_mean = np.mean(dist_err_mm, axis=0)
    dist_err_mm_landmarks_std = np.std(dist_err_mm, axis=0)
    # Overall mean and std of distance error
    dist_err_mean = np.mean(dist_err)
    dist_err_std = np.std(dist_err)
    dist_err_mm_mean = np.mean(dist_err_mm)
    dist_err_mm_std = np.std(dist_err_mm)

    with open(os.path.join(save_dir, 'dist_error_'+suffix+'.txt'), 'w') as f:
        # Record results in pixel
        f.write("Distance error (pixel)\n")
        for i in xrange(img_count):
            f.write("{} ".format(names[i]))
            for j in xrange(dist_err.shape[1]):
                f.write("{:.10f} ".format(dist_err[i, j]))
            f.write("\n")
        f.write("\nMean: ")
        for i in xrange(num_landmarks):
            f.write("{} ".format(dist_err_landmarks_mean[i]))
        f.write("\nStandard deviation: ")
        for i in xrange(num_landmarks):
            f.write("{} ".format(dist_err_landmarks_std[i]))
        f.write("\nMean: {:.10f}".format(dist_err_mean))
        f.write("\nStandard deviation: {:.10f}".format(dist_err_std))

        # Record results in mm
        f.write("\n\nDistance error (mm)\n")
        for i in xrange(img_count):
            f.write("{} ".format(names[i]))
            for j in xrange(dist_err_mm.shape[1]):
                f.write("{:.10f} ".format(dist_err_mm[i, j]))
            f.write("\n")
        f.write("\nMean: ")
        for i in xrange(num_landmarks):
            f.write("{} ".format(dist_err_mm_landmarks_mean[i]))
        f.write("\nStandard deviation: ")
        for i in xrange(num_landmarks):
            f.write("{} ".format(dist_err_mm_landmarks_std[i]))
        f.write("\nMean: {:.10f}".format(dist_err_mm_mean))
        f.write("\nStandard deviation: {:.10f}".format(dist_err_mm_std))


def save_landmarks(save_dir, train, names, landmarks, landmark_unwant):
    """Save as txt file the predicted landmarks.

    Args:
      save_dir: Directory storing the results.
      train: train or test dataset
      names: list of names of the patients.
      landmarks: 3D numpy array of the predicted landmarks. [img_count, num_landmarks, 3]
      landmark_unwant: list of indices of unwanted landmarks

    """
    if train:
        save_dir = os.path.join(save_dir, 'train')
    else:
        save_dir = os.path.join(save_dir, 'test')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    landmark_count_original = landmarks.shape[1] + len(landmark_unwant)
    for i in range(landmarks.shape[0]):
        add_mask = np.zeros(landmark_count_original, dtype=bool)
        add_ind = list(landmark_unwant)
        add_mask[add_ind] = True
        counter = 0

        with open(os.path.join(save_dir, names[i] + '_ps.txt'), 'w') as f:
            for j in range(landmark_count_original):
                str = ""
                if add_mask[j]:
                    for k in range(landmarks.shape[2]):
                        str += (" {:.10f}".format(0.0))
                else:
                    for k in range(landmarks.shape[2]):
                        str += ("{:.10f} ".format(landmarks[i, counter, k]))
                    counter += 1
                str += "\n"
                f.write(str)
