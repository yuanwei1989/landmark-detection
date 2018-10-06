"""Functions for plane manipulations."""

import numpy as np
import scipy.ndimage


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def fit_line(pts):
    """Fit a line to a set of 3D points.

    Args:
      pts: [point_count, 3]

    Returns:
      d: direction vector of line [3]
      c: a point on the line [3]

    """
    c = pts.mean(axis=0)
    A = pts - c
    u, s, vh = np.linalg.svd(A)
    d = vh[0, :]
    # ensure x-component of direction vector is always consistent (eg. positive)
    if d[0] < 0:
        d = -d
    return d, c


def fit_plane(pts):
    """Fit a plane to a set of 3D points.

    Args:
      pts: [point_count, 3]

    Returns:
      n: normal vector of plane [3]
      c: centroid of plane [3]

    """
    c = pts.mean(axis=0)
    A = pts - c
    u, s, vh = np.linalg.svd(A)
    n = vh[-1, :]
    # ensure z-component of normal vector is always consistent (eg. positive)
    if n[2] < 0:
        n = -n
    return n, c

def project_on_plane(pts, n, c):
    """Project points onto a 2D plane.

    Args:
      pts: [point_count, 3]
      n: normal vector of plane [3]
      c: centroid of plane [3]

    Returns:
      pts_new: points projected onto the plane [point_count, 3]

    """
    t = (np.dot(c, n) - np.dot(pts, n)) / np.dot(n, n)
    pts_new = pts + np.matmul(np.expand_dims(t, axis=1), np.expand_dims(n, axis=0))
    return pts_new


def extract_tform(landmarks, plane_name):
    """Compute 4x4 transformation matrix that map the reference xy-plane at origin to the TV/TC plane defined by landmarks.

    Args:
      landmarks: TV plane landmarks. [landmark_count, 3]
      plane_name: 'tv' or 'tc'

    Returns:
      mat: 4x4 transformation matrix [4, 4]

    """
    if plane_name=='tv':
        # Fit plane and project landmarks onto plane
        z_vec, p_plane = fit_plane(landmarks)
        landmarks_proj = project_on_plane(landmarks, z_vec, p_plane)

        # Fit mid line
        landmarks_line = landmarks_proj[[0,1,2,7], :]
        x_vec, p_line = fit_line(landmarks_line)
        y_vec = unit_vector(np.cross(z_vec, x_vec))

        # 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, :3] = np.vstack((x_vec, y_vec, z_vec)).transpose()
        mat[:3, 3] = landmarks_proj[0]

    elif plane_name=='tc':
        # Landmarks lying on the TC plane
        cr = landmarks[0]
        cl = landmarks[1]
        csp = landmarks[2]

        # Orthogonal basis
        csp_cl = cl - csp
        csp_cr = cr - csp
        z_vec = np.cross(csp_cl, csp_cr)
        z_vec = unit_vector(z_vec)
        cr_cl_mid = (cr + cl) / 2.0
        x_vec = unit_vector(cr_cl_mid - csp)
        y_vec = unit_vector(np.cross(z_vec, x_vec))

        # 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, :3] = np.vstack((x_vec, y_vec, z_vec)).transpose()
        mat[:3, 3] = (cr_cl_mid + csp) / 2.0

    else:
        raise ValueError('Invalid plane name.')

    return mat


def init_mesh(mesh_siz):
    """Initialise identity plane with a fixed size

        Args:
            mesh_siz: size of plane. Odd number only. [2]

        Returns:
            xyz_coords: mesh coordinates of identity plane. [4, num_mesh_points]

    """
    mesh_r = (mesh_siz - 1) / 2
    x_lin = np.linspace(-mesh_r[0], mesh_r[0], mesh_siz[0])
    y_lin = np.linspace(-mesh_r[1], mesh_r[1], mesh_siz[1])
    xy_coords = np.meshgrid(y_lin, x_lin)
    xyz_coords = np.vstack([xy_coords[1].reshape(-1),
                            xy_coords[0].reshape(-1),
                            np.zeros(mesh_siz[0] * mesh_siz[1]),
                            np.ones(mesh_siz[0] * mesh_siz[1])])
    return xyz_coords


def extract_plane_from_mesh(image, mesh, mesh_siz, order):
    """Extract a 2D plane image from the 3D volume given the mesh coordinates of the plane.

        Args:
          image: 3D volume. [x,y,z]
          mesh: mesh coordinates of a plane. [4, num_mesh_points]. Origin at volume centre
          mesh_siz: size of mesh [2]
          order: interpolation order (0-5)

        Returns:
          slice: 2D plane image [plane_siz[0], plane_siz[1]]
          xyz_coords_new: mesh coordinates of the plane. Origin at volume corner. numpy array of size [3, plane_siz[0], plane_siz[1]

    """
    # Set image matrix corner as origin
    img_siz = np.array(image.shape)
    img_c = (img_siz-1)/2.0
    mesh_new = mesh[:3, :] + np.expand_dims(img_c, axis=1)

    # Reshape coordinates
    x_coords = mesh_new[0, :].reshape(mesh_siz)
    y_coords = mesh_new[1, :].reshape(mesh_siz)
    z_coords = mesh_new[2, :].reshape(mesh_siz)
    xyz_coords_new = np.stack((x_coords, y_coords, z_coords), axis=0)

    # Extract image plane
    slice = scipy.ndimage.map_coordinates(image, xyz_coords_new, order=order)
    return slice, xyz_coords_new


def extract_plane_from_pose(image, mat, plane_siz, order):
    """Extract a 2D plane image from the 3D volume given the pose (transformation matrix) wrt the identity plane.

        Args:
          image: 3D volume. [x,y,z]
          mat: 4x4 transformation matrix [4, 4]
          plane_siz: size of plane [2]
          order: interpolation order (0-5)

        Returns:
          slice: 2D plane image [plane_siz[0], plane_siz[1]]
          xyz_coords_new: mesh coordinates of the plane. Origin at volume corner. numpy array of size [3, plane_siz[0], plane_siz[1]]

    """
    # Initialise identity plane
    xyz_coords = init_mesh(plane_siz)

    # Rotate and translate plane
    xyz_coords = np.dot(mat, xyz_coords)

    # Extract image plane
    slice, xyz_coords_new = extract_plane_from_mesh(image, xyz_coords, plane_siz, order)

    return slice, xyz_coords_new
