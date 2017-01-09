from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass

import numpy as np

def least_squares_homography(points1, points2):
  """
  Computes homography transforming points1 towards points2.
  Uses least squares method
  input:
    points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    points1[i,:] corresponds to poins2[i,:]
  Returns:
    A 3X3 array with the computed homography.
    In case of instable solutions returns None.
  """
  p1, p2 = points1, points2
  o0, o1 = np.zeros((p1.shape[0],1)), np.ones((p1.shape[0],1))

  A = np.vstack([np.hstack([p1[:,:1], o0, -p1[:,:1]*p2[:,:1], p1[:,1:],o0, -p1[:,1:]*p2[:,:1],o1,o0]),
                 np.hstack([o0, p1[:,:1],-p1[:,:1]*p2[:,1:],o0,p1[:,1:],-p1[:,1:]*p2[:,1:],o0,o1])])

  # Return None for unstable solutions
  if np.linalg.matrix_rank(A, 1e-3) < 8:
    return None
  if A.shape[0] == 8 and np.linalg.cond(A) > 1e10:
    return None

  H = np.linalg.lstsq(A, p2.T.flatten())[0]
  H = np.r_[H,1]
  return H.reshape((3,3)).T

def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  input:
    image: 2D image
  Returns:
    A boolean array with the same shape as image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image).astype(np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret

def spread_out_corners(im, m, n, radius):
  """
  Split the image im to m by n rectangles and use harris_corner_detector on each.
  input:
    im -- 2D array representing an image.
    m -- vertical number of rectangles.
    n -- horizontal number of rectangles.
    radius -- Minimal distance of corner points from the borders of the image.
   Returns:
    An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  from sol4 import harris_corner_detector
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) & 
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  return corners[legit,:]

