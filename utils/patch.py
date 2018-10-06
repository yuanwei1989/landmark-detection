"""Functions for patch extraction."""

import numpy as np


def extract_patch(image, xc, yc, box_r):
    """Extract a patch from image given the centre (xc,yc) of patch and the radius of patch (box_size).
       Out of image region is set to zeros.

        Args:
          image: [width, height, channel].
          xc, yc: current patch centre coordinates,
          box_r: radii of image patch in x, y axis

        Returns:
          patch: [box_size, box_size, channel]

    """
    xc = int(xc)
    yc = int(yc)
    box_r = int(box_r)
    max_x = image.shape[0]
    max_y = image.shape[1]

    # Obtain box coordinates
    x_start = xc - box_r
    x_end = xc + box_r + 1
    y_start = yc - box_r
    y_end = yc + box_r + 1

    # Pad zeros for out of image region
    x_start_pad = 0
    x_end_pad = 0
    y_start_pad = 0
    y_end_pad = 0
    if (x_end<=0 or x_start>=max_x or y_end<=0 or y_start>=max_y):
        return np.zeros((box_r*2+1, box_r*2+1, image.shape[2]))
    if x_start < 0:
        x_start_pad = -x_start
        x_start = 0
    if x_end > max_x:
        x_end_pad = x_end - max_x
        x_end = max_x
    if y_start < 0:
        y_start_pad = -y_start
        y_start = 0
    if y_end > max_y:
        y_end_pad = y_end - max_y
        y_end = max_y

    # Extract image patch
    patch = image[x_start:x_end, y_start:y_end, :]
    patch = np.lib.pad(patch, ((x_start_pad, x_end_pad), (y_start_pad, y_end_pad), (0, 0)), 'constant')

    return patch


def extract_patch_all_planes(image, xc, yc, zc, box_r):
    """Extract a patch from all three orthogonal planes given the centre (xc,yc,zc) of patch and the radius of patch (box_xr, box_yr, box_zr).
       Out of image region is set to zeros.

        Args:
          image: [width, height, depth, channel].
          xc, yc, zc: current patch centre coordinates,
          box_r: radii of image patch in x, y, z axis

        Returns:
          patch: [box_size, box_size, channel]

    """
    xc = int(xc)
    yc = int(yc)
    zc = int(zc)
    box_size = box_r * 2 +1

    # yz plane
    if (xc < 0) or (xc >= image.shape[0]):
        yz_patch = np.zeros((box_size, box_size, 1))
    else:
        yz_patch = extract_patch(image[xc, :, :, :], yc, zc, box_r)

    # xz plane
    if (yc < 0) or (yc >= image.shape[1]):
        xz_patch = np.zeros((box_size, box_size, 1))
    else:
        xz_patch = extract_patch(image[:, yc, :, :], xc, zc, box_r)

    # xy plane
    if (zc < 0) or (zc >= image.shape[2]):
        xy_patch = np.zeros((box_size, box_size, 1))
    else:
        xy_patch = extract_patch(image[:, :, zc, :], xc, yc, box_r)

    patch = np.concatenate((yz_patch,               # yz plane
                            xz_patch,               # xz plane
                            xy_patch), axis=2)      # xy plane
    return patch


def extract_patch_all_landmarks(image, landmarks, box_r):
    """Extract patches for all the landmarks in a volume. Out of image region is set to zeros.
    Each landmark has patches of dimension=[box_r box_r 3]. Return output has dimension=[box_r box_r 3*num_landmarks]

        Args:
          image: [width, height, depth, channel].
          landmarks: [num_landmarks, 3],
          box_r: radii of image patch in x, y, z axis

        Returns:
          patches: [box_size, box_size, 3*num_landmarks]

    """
    box_size = box_r*2+1
    patches = np.empty((box_size, box_size, 0))
    for i in xrange(landmarks.shape[0]):
        patch = extract_patch_all_planes(image, landmarks[i, 0], landmarks[i, 1], landmarks[i, 2], box_r)
        patches = np.concatenate((patches, patch), axis=2)
    return patches
