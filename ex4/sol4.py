import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates

from ex4 import sol4_add
from ex4 import  sol4_utils



from random import sample
import math


# __________________________________________________________________________________________

# ================================= constants =================================
X_DER_KERNEL = np.array([1, 0, -1])[np.newaxis]
Y_DER_KERNEL = np.transpose(X_DER_KERNEL)

BLUR_KERNEL_SIZE = 3
K_RESPONSE = 0.04

DESC_PYR_LEVELS = 3

SPREAD_CORNERS_M = 7  # todo: set params
SPREAD_CORNERS_N = 7  # todo: set params
SPREAD_CORNERS_RAD = 10  # todo: set params

SAMPLE_DESC_RAD = 3

HOMOGRAPHY_N_POINTS = 4

N_COORS = 3

GRAYSCALE = 1
RGB = 2
MAX_GRAYSCALE_INTENSITY = 255

BASIC_GAUSSIAN_FILTER_VEC = np.array([1, 1])[np.newaxis]
MIN_PYR_LEVEL_DIM = 16
SAMPLE_RATE = 2
ROWS = 0
COLS = 1


# ============================== private functions ============================
# def _get_level_coors(pos, from_level, to_level):
#     """
#     Converts the given point's coordinates from the pyramid's from_level to the pyramid's to_level
#     :param pos: The coordinates of the points in the from_level (assumes pos.shape == (N, 2))
#     :param from_level: The pyramid's level from which the coordinates are converted
#     :param to_level: The pyramid's level to which the coordinates are converted
#     :return: The coordinates of the given points in the to_level
#     """
#     levels_factor = (SAMPLE_RATE ** (from_level - to_level))
#     return pos * [levels_factor, levels_factor]


def _get_indices_offsets(rad):
    """
    :param rad: The radius of the indices offsets
    :return: A (2, d) array (where d = 2 *rad + 1) contains all the indices offsets with respect to
             the given radius: [-rad, -rad], [-rad, -rad+1], ..., [rad, rad-1], [rad, rad].
    """
    diam = rad * 2 + 1
    # print(np.indices((diam, diam)).reshape(2, -1))
    return np.indices((diam, diam)).reshape(2, -1).T - rad


def _normalize_cols(matrix):
    """
    :param matrix: The input matrix
    :return: The Columns normalized input matrix
    """
    row_means = np.mean(matrix, axis=0)
    print(row_means.shape)
    print(row_means[1])
    norm_factors = np.linalg.norm(matrix, axis=0)
    print(norm_factors[1])
    norm_factors[norm_factors == 0] = 1
    return (matrix - row_means) / norm_factors


def _calc_2nd_max_rows_mask(matrix):
    """
    This function doesn't change the input matrix
    :param matrix: The input matrix
    :return: A mask of the same matrix's shape, where it's (i,j) element is true iff matrix[i,j] not
             smaller than the second maximum of the matrix's i'th row
    """
    matrix_copy = np.copy(matrix)
    rows_max = np.max(matrix_copy, axis=1)[:, np.newaxis]
    max_inds = matrix_copy == rows_max
    matrix_copy[max_inds] = -np.inf
    rows_2nd_max = np.max(matrix_copy, axis=1)[:, np.newaxis]
    return matrix >= rows_2nd_max


def _plot_lines(points1, points2, color):
    """
    Plots lines in the specified color between points1[i] and points2[i] for i=0,...,N-1
    :param points1: (N,2) array of points
    :param points2: (N,2) array of points
    :param color: The color of the lines
    """
    points_dstack = np.dstack((points1, points2))
    for (plot_ys, plot_xs) in points_dstack:
        plt.plot(plot_xs, plot_ys, mfc='r', c=color, lw=.4, ms=5,
                 marker='o')  # todo: check params (changes ms from school)


def _get_panorama_boundaries(ims, Hs):
    """
    :param ims: The panorama's images
    :param Hs: The panorama's homographies
    :return: pan_boundaries: The panorama's coordinate boundaries: min_x, max_x, min_y, max_y, calculated
                             in the panorama coordinate system (meaning that the boundaries are calculated
                             after the corresponding homography is applied on it's corresponding image,
                             and therefore the boundaries might be negative for example)
             pan_strips_boundaries: An array of shape (n_ims, 2), which contains the panorama's strips
                                    boundaries (the i'th row corresponds to the i'th strip's boundaries)
            pan_masks_boundaries: An array of shape (n_ims-1,), the i'th element is the boundary between
                                  the i'th and the  i+1'th images
    """
    # getting images corners
    n_ims = len(ims)
    ims_corners = []
    ims_centers = []
    for im in ims:
        n_rows, n_cols = im.shape
        im_corners = np.array([[0, 0], [n_rows - 1, 0], [0, n_cols - 1],
                               [n_rows - 1, n_cols - 1]])
        ims_corners.append(im_corners)
        ims_center = np.array([n_rows // 2, n_cols // 2])
        ims_centers.append(ims_center)

    # calculating the corners in the panorama coordinate system
    pan_corners = [apply_homography(im_corners, H) for (im_corners, H) in
                   zip(ims_corners, Hs)]
    pan_corners_flat = np.vstack(pan_corners)
    pan_corners_xs = pan_corners_flat[:, 1]
    pan_corners_ys = pan_corners_flat[:, 0]

    # calculating the min and max of each axis of the panorama
    min_x, max_x = int(floor(np.min(pan_corners_xs))), int(
        ceil(np.max(pan_corners_xs)))  # todo: check calc
    min_y, max_y = int(floor(np.min(pan_corners_ys))), int(
        ceil(np.max(pan_corners_ys)))  # todo: check calc
    pan_boundaries = min_x, max_x, min_y, max_y

    # calculating the strips boundaries
    pan_corners_xs_matrix = pan_corners_xs.reshape(n_ims, 4)
    pan_left_boundaries = np.min(pan_corners_xs_matrix[:, [0, 1]], axis=1)
    pan_right_boundaries = np.max(pan_corners_xs_matrix[:, [2, 3]], axis=1)
    pan_strips_boundaries = np.vstack(
        (pan_left_boundaries, pan_right_boundaries)).T.astype(np.int64)

    # calculating the masks boundaries
    pan_x_centers = [apply_homography(im_center[np.newaxis], H)[0][1] for
                     (im_center, H) in zip(ims_centers, Hs)]
    pan_masks_boundaries = [int((pan_x_centers[i] + pan_x_centers[i + 1]) // 2)
                            for i in range(n_ims - 1)]

    return pan_boundaries, pan_strips_boundaries, pan_masks_boundaries


def _get_indices_in_ranges(min_x, max_x, min_y, max_y):
    """
    :param min_x: min x value
    :param max_x: max x value
    :param min_y: min y value
    :param max_y: max y value
    :return: Array of shape (N, 2) where N = (max_x - min_x) * (max_y - min_y), containing all the
             indices in the given ranges (x range from min_x to max_x and y range from min_y to max_y)
    """
    x_range = range(min_x, max_x)
    y_range = range(min_y, max_y)

    x_matrix_inds, y_matrix_inds = np.meshgrid(x_range, y_range)
    x_inds_flat = np.hstack(x_matrix_inds)
    y_inds_flat = np.hstack(y_matrix_inds)
    return np.vstack((y_inds_flat, x_inds_flat)).T


def _min_cut_path(error_strip):
    """
    :param error_strip: The error strip array
    :return: An array of shape (N,) where N is the number of error_strip's rows, which is
             the minimal cut's path considering max of 1 column step (right/left) at each row
    """
    n_rows, n_cols = error_strip.shape
    acc_error = np.zeros_like(error_strip, dtype=np.float64)  # todo check 32
    pointers = np.zeros_like(error_strip, dtype=np.uint64)  # todo check 32

    # calculating the accumulative error's table and the pointers table
    cols_inds = np.arange(n_cols, dtype=np.int64)  # todo check 32
    acc_error[0, :] = error_strip[0, :]
    for i in range(1, n_rows):
        prev_left_acc_error = np.insert(acc_error[i - 1, :-1], 0, [np.inf])
        prev_center_acc_error = acc_error[i - 1, :]
        prev_right_acc_error = np.append(acc_error[i - 1, 1:], [np.inf])
        prev_error = np.vstack(
            (prev_left_acc_error, prev_center_acc_error, prev_right_acc_error))
        prev_arg_min = np.argmin(prev_error, axis=0)
        prev_min_error = np.min(prev_error, axis=0)  # todo may optimize

        pointers[i, :] = cols_inds + prev_arg_min - 1
        acc_error[i, :] = error_strip[i, :] + prev_min_error

    # constructing the optimal path (from the last row to the first row)
    min_path = np.zeros(n_rows, dtype=np.int64)
    cur_arg_min = np.argmin(acc_error[-1, :])
    min_path[-1] = cur_arg_min
    for i in range(1, n_rows):
        cur_arg_min = pointers[-i, cur_arg_min]
        min_path[-i - 1] = cur_arg_min

    return min_path


def _get_min_cut_mask(overlap_left_boundary, overlap_right_boundary, panorama,
                      pan_strip):
    """
    :param overlap_left_boundary: overlap strip's left boundary
    :param overlap_right_boundary: overlap strip's right boundary
    :param panorama: The panorama image
    :param pan_strip: The panorama's strip image
    :return: An array of the same shape as the given panorama, which is the minimal cut mask
             (the mask is induced from the minimal cut path in the given overlap strip boundaries)
    """
    error_strip = np.square(
        panorama[:, overlap_left_boundary:overlap_right_boundary] -
        pan_strip[:, overlap_left_boundary:overlap_right_boundary])
    min_cut_rel_path = _min_cut_path(error_strip)
    min_cut_path = min_cut_rel_path + overlap_left_boundary
    x_inds_T = np.indices(panorama.shape)[1].T
    return (x_inds_T < min_cut_path).T


def _calc_im_strip(pan_strip_left_boundary, pan_strip_right_boundary, min_y,
                   max_y, im, H):
    """
    :param pan_strip_left_boundary: panorama's strip's left boundary (in panorama system coordinates)
    :param pan_strip_right_boundary: panorama's strip's right boundary (in panorama system coordinates)
    :param min_y: panorama's min y (in panorama system coordinates)
    :param max_y: panorama's max y (in panorama system coordinates)
    :param im: The strip's image
    :param H: The homography which corresponds to the given image
    :return: A strip of the panorama, contains the homographed image
    """
    strip_n_cols = pan_strip_right_boundary - pan_strip_left_boundary
    pan_strip_points = _get_indices_in_ranges(pan_strip_left_boundary,
                                              pan_strip_right_boundary, min_y,
                                              max_y)  # todo +1 because comment above (PLUS_ONE_SEARCH)
    im_strip_points = apply_homography(pan_strip_points, np.linalg.inv(H))

    im_strip_flat = map_coordinates(im, im_strip_points.T, order=1,
                                    prefilter=False)
    return im_strip_flat.reshape((max_y - min_y,
                                  strip_n_cols))  # todo +1 because comment above (PLUS_ONE_SEARCH)


# ============================== public functions =============================
# def harris_corner_detector(im):
#     """
#     :param im: grayscale image to find key points inside
#     :return: An array with shape (N,2) of [x,y] key points locations in im
#     """
#     x_der = convolve(im, X_DER_KERNEL)
#     y_der = convolve(im, Y_DER_KERNEL)
#
#     # blurring the derivatives matrices
#     x_der_2 = blur_spatial(np.square(x_der), BLUR_KERNEL_SIZE)
#     y_der_2 = blur_spatial(np.square(y_der), BLUR_KERNEL_SIZE)
#     x_der_y_der = blur_spatial(x_der * y_der, BLUR_KERNEL_SIZE)
#
#     # calculating the matrices's determinants and traces
#     det_m = x_der_2 * y_der_2 - (np.square(x_der_y_der))
#     trace_m = x_der_2 + y_der_2
#
#     # calculating the response image
#     response_im = det_m - K_RESPONSE * (np.square(trace_m))
#     local_max_response_im = non_maximum_suppression(response_im)
#
#     # swap rows and cols to sustain coordinate convention
#     return np.argwhere(local_max_response_im)[:, [1, 0]]


# def sample_descriptor(im, pos, desc_rad):
#
#     # calculating the descriptor's diameter and size
#     desc_diam = 2 * desc_rad + 1
#     desc_size = desc_diam ** 2
#
#     # switching to rows and cols convention
#     points = pos[:, [1, 0]]
#
#     # repeating each point descriptor's size times
#     points_rep = np.repeat(points, desc_size, axis=0)
#
#     # repeating each indices offsets the number of positions times
#     len_points = len(points)
#     indices_off = _get_indices_offsets(desc_rad)
#     indices_off_rep = np.tile(indices_off, (len_points, 1))
#
#     # calculating the descriptor offsets points
#     points_off = points_rep + indices_off_rep
#
#     # acquiring the normalized descriptors
#     descs_flat = map_coordinates(im, points_off.T, order=1, prefilter=False)
#     descs_cols = descs_flat.reshape(len_points, desc_size).T
#     norm_descs_cols = _normalize_cols(descs_cols)
#     return norm_descs_cols.reshape(desc_diam, desc_diam, len_points)


# def find_features(pyr):
#     """
#     :param pyr: Gaussian pyramid of a grayscale image having 3 levels
#     :return: pos: An array with shape (N,2) of [x,y] feature location per row found in the
#                   (third pyramid level of the) image. These coordinates are provided at the pyramid
#                   level pyr[0]
#              desc: A feature descriptor array with shape (K,K,N), sampled at the top pyramid's level
#     """
#     pyr_top_level_ind = DESC_PYR_LEVELS - 1
#     pos = sol4_add.spread_out_corners(pyr[0], SPREAD_CORNERS_M, SPREAD_CORNERS_N, SPREAD_CORNERS_RAD)
#     pos_top_level = _get_level_coors(pos, 0, pyr_top_level_ind)
#     desc = sample_descriptor(pyr[pyr_top_level_ind], pos_top_level, SAMPLE_DESC_RAD)
#     return pos, desc


# def match_features(desc1, desc2, min_score):
#     """
#     :param desc1: A feature descriptor array with shape (K,K,N1)
#     :param desc2: A feature descriptor array with shape (K,K,N2)
#     :param min_score: Minimal match score between two descriptors required to be regarded as
#                       corresponding points
#     :return: match_ind1: Array with shape (M,) and dtype int of matching indices in desc1
#              match_ind1: Array with shape (M,) and dtype int of matching indices in desc2
#     """
#     n1, n2 = desc1.shape[-1], desc2.shape[-1]
#     desc_len = desc1.shape[0] * desc1.shape[1]
#
#     # reshaping the descriptors to a matrix where each column corresponds to descriptor
#     desc1 = desc1.reshape((desc_len, n1))
#     desc2 = desc2.reshape((desc_len, n2))
#
#     # matrices whose (i,j) element equals to the dot product of desc1[:, i] and desc[:, j]
#     matches_scores = np.dot(desc1.T, desc2)
#
#     for t in [.2,.3,.4,.5]:     #todo rm
#         print(str(t) + ':', np.sum(matches_scores > t))     #todo rm
#     print(np.min(matches_scores), np.max(matches_scores))   #todo rm
#
#     # finds the matches which isn't smaller than the 2nd max of each axis and is above min_score
#     second_max_rows = _calc_2nd_max_rows_mask(matches_scores)
#     second_max_cols = _calc_2nd_max_rows_mask(matches_scores.T).T
#     above_min_score = matches_scores > min_score
#
#     matches_inds_matrix = second_max_rows & second_max_cols & above_min_score
#     matches_inds = np.argwhere(matches_inds_matrix)
#
#     return matches_inds[:, 0], matches_inds[:, 1]


# def apply_homography1(pos1, H12):
#     """
#     :param pos1: An array with shape (N,2) of [x,y] point coordinates
#     :param H12: A 3x3 homography matrix
#     :return: An array with the same shape as pos1 with [x,y] point coordinates in image i+1 obtained
#              from transforming pos1 using H12
#     """
#     # adding 1 to 3rd coordinate to maintain homogeneous coordinates
#     hom_pos1_T = np.hstack((pos1, np.ones((pos1.shape[0], 1)))).T
#     non_hom_pos2_T = np.dot(H12, hom_pos1_T)
#     non_hom_pos2_T[-1, :][non_hom_pos2_T[-1, :] == 0] = 1e-100  # todo rm
#     pos2_T = non_hom_pos2_T[:-1, :] / non_hom_pos2_T[-1,
#                                       :]  # todo check for 0 division
#     return pos2_T.T


# def ransac_homography(pos1, pos2, num_iters, inlier_tol):
#     """
#     :param pos1: Array with shape (N,2) containing n rows of [x,y] coordinates of matched points
#     :param pos2: Array with shape (N,2) containing n rows of [x,y] coordinates of matched points
#     :param num_iters: Number of RANSAC iterations to perform
#     :param inlier_tol: inlier tolerance threshold
#     :return: max_hom: A 3x3 normalized homography matrix
#              max_inliers_inds : An Array with shape (S,) where S is the number of inliers,
#                                 containing the indices in pos1/pos2 of the maximal set of inlier
#                                 matches found
#     """
#     assert (len(pos1) == len(pos2))  # todo rm
#     assert (len(pos1) > 4)  # todo rm
#
#     n_pos = len(pos1)
#     indices = range(n_pos)
#
#     # keeping the largest set of inliers (and the size of this max set)
#     max_inliers_inds = None
#     max_n_inliers = -np.inf
#
#     s, f = 0, 0  # todo rm
#
#     for i in range(num_iters):
#         # calculating homography based on random sampled points
#         random_indices = sample(indices, HOMOGRAPHY_N_POINTS)
#         p1, p2 = pos1[random_indices], pos2[random_indices]
#         hom = sol4_add.least_squares_homography(p1, p2)
#         if hom is None:
#             f += 1  # todo rm
#             continue
#         s += 1  # todo rm
#
#         pos2_tag = apply_homography(pos1, hom)
#         error = np.sum(np.square(pos2_tag - pos2), axis=1)
#
#         inliers_inds = error < inlier_tol
#         n_inliers = np.sum(inliers_inds)
#         if n_inliers > max_n_inliers:
#             max_inliers_inds = inliers_inds
#             max_n_inliers = n_inliers
#
#     print('s =', s)  # todo rm
#     print('f =', f)  # todo rm
#
#     # calculating the final homography based on the largest set of inliers points
#     max_p1, max_p2 = pos1[max_inliers_inds, :], pos2[max_inliers_inds, :]
#     max_hom = sol4_add.least_squares_homography(max_p1, max_p2)
#     return max_hom, max_inliers_inds


# def display_matches1(im1, im2, pos1, pos2, inliers):
#     """
#     :param im1: grayscale image
#     :param im2: grayscale image
#     :param pos1: Array with shape (N,2), containing N rows of [x,y] coordinates of matched points
#                  in im1
#     :param pos2: Array with shape (N,2), containing N rows of [x,y] coordinates of matched points
#                  in im1
#     :param inliers: An array with shape (S,) of inlier matches
#     """
#     images = np.hstack((im1, im2))
#
#     im2_shift = im1.shape[1]
#     pos2[:, 1] += im2_shift
#
#     # plotting images
#     plt.figure()
#     plt.imshow(images, cmap=plt.cm.gray)
#
#     # splitting points to inliers and outliers
#     outliers = np.logical_not(inliers)
#     pos1_inliers, pos1_outliers = pos1[inliers], pos1[outliers]
#     pos2_inliers, pos2_outliers = pos2[inliers], pos2[outliers]
#
#     # plotting inliers and outliers matches
#     _plot_lines(pos1_inliers, pos2_inliers, 'yellow')
#     _plot_lines(pos1_outliers, pos2_outliers, 'blue')


# def accumulate_homographies2(H_successive, m):
#     # splitting the Hs to left and inverse right Hs
#     left_Hs = H_successive[:m]
#     right_Hs = H_successive[m:]
#     inv_right_Hs = [np.linalg.inv(right_H) for right_H in right_Hs]
#
#     # inserting the left Hs to the left and the inverse right to the right of H2m
#     H2m = [np.eye(N_COORS)]
#     for left_H in left_Hs[::-1]:
#         H2m.insert(0, np.dot(H2m[0], left_H))
#     for inv_right_H in inv_right_Hs:
#         H2m.append(np.dot(H2m[-1], inv_right_H))
#
#     # normalizing according to the homographies element at index (2,2)
#     H2m = [H / H[N_COORS - 1, N_COORS - 1] for H in H2m]
#     return H2m


# todo: tmp rm
m = 0
ma = None


def render_panorama(ims, Hs):
    """
    :param ims: A list of grayscale images
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from
               the coordinate system of ims [i] to the coordinate system of the panorama
    :return: A grayscale panorama image composed of vertical strips, backwarped using homographies
             from Hs, one from every image in ims
    """
    pan_boundaries, pan_strips_boundaries, pan_masks_boundaries = _get_panorama_boundaries(
        ims, Hs)
    min_x, max_x, min_y, max_y = pan_boundaries

    if min_x % 2:  # todo rm (should affect only blending not entrire pan)
        min_x -= 1  # todo rm
    if max_x % 2:  # todo rm
        max_x += 1  # todo rm
    if min_y % 2:  # todo rm
        min_y -= 1  # todo rm
    if max_y % 2:  # todo rm
        max_y += 1  # todo rm

    pan_n_rows = max_y - min_y  # todo: if rounding in min,max then +1 is redundant (but if not rounding how shouold use meshgrid?), if removing, serach in code for "PLUS_ONE_SEARCH"
    pan_n_cols = max_x - min_x

    panorama = np.zeros((pan_n_rows, pan_n_cols))
    n_ims = len(ims)
    for i in range(n_ims):
        pan_strip_left_boundary, pan_strip_right_boundary = \
        pan_strips_boundaries[i]
        im_strip = _calc_im_strip(pan_strip_left_boundary,
                                  pan_strip_right_boundary, min_y, max_y,
                                  ims[i], Hs[i])

        # switching to real panorama image coordinates
        left_boundary, right_boundary = pan_strip_left_boundary - min_x, pan_strip_right_boundary - min_x

        # the leftmost first image
        if i == 0:
            panorama[:, left_boundary:right_boundary] = im_strip
            continue

        pan_strip = np.zeros_like(panorama)
        pan_strip[:, left_boundary:right_boundary] = im_strip

        # mask = np.zeros_like(panorama, dtype=np.bool)
        # mask_boundary = pan_masks_boundaries[i-1] - min_x
        # mask[:, :mask_boundary] = True

        # todo: bonus impl (question in forum about diff masks for diff colors)
        overlap_left_boundary, overlap_right_boundary = left_boundary, \
                                                        pan_strips_boundaries[
                                                            i - 1][1] - min_x
        mask = _get_min_cut_mask(overlap_left_boundary, overlap_right_boundary,
                                 panorama, pan_strip)

        # # todo rm
        # plt.figure()
        # plt.subplot(221)
        # plt.title('pan')
        # plt.imshow(panorama, cmap=plt.cm.gray)
        # plt.subplot(222)
        # plt.title('strip')
        # plt.imshow(pan_strip, cmap=plt.cm.gray)
        # plt.subplot(223)
        # plt.title('maks')
        # plt.imshow(mask, cmap=plt.cm.gray)
        # plt.figure()
        # blend = panorama * .5 + pan_strip * .5 + mask * .2
        # plt.imshow(blend, cmap=plt.cm.gray)
        # plt.show()

        # panorama[:, left_boundary:right_boundary] = im_strip
        panorama = sol4_utils.pyramid_blending(panorama, pan_strip, mask, 2, 9, 9)

    return panorama


# # todo: remove, used for school script
# def im_to_points(im):
#     pyr = build_gaussian_pyramid(im, 3, 3)[0]
#     pos, desc = find_features(pyr)
#     points = pos[:, [1, 0]]
#     return points, desc


# _____________________________________________________________________________________________
DER_VEC = np.array([1, 0, -1])[np.newaxis]
DEFAULT_KER_SIZE = 3
DEFAULT_K = 0.04
FIRST_EIG_VAL_IND = 0
SEC_EIG_VAL_IND = 0
ORIG_IM = 0
DEFAULT_DESC_RAD = 3
DEFAULT_RADIUS = 10
DEF_N = 7
DEF_M = 7

def harris_corner_detector(im):
    '''
    Finding haris point of interest
    :param im: grayscale image to find key points inside
    :return: An array with shape (N,2) of [x,y] key points locations in im
    '''
    # derX1, derY2 = conv_der(im)
    # print(derX1)
    derX = convolve(im, DER_VEC)
    derY = convolve(im, np.transpose(DER_VEC))
    # print(derX1 == derX)

    bluredSquaredDerX = sol4_utils.blur_spatial(np.multiply(derX, derX), DEFAULT_KER_SIZE)
    bluredSquaredDerY = sol4_utils.blur_spatial(np.multiply(derY, derY), DEFAULT_KER_SIZE)
    bluredDerXderY = sol4_utils.blur_spatial(np.multiply(derX, derY), DEFAULT_KER_SIZE)

    det = np.multiply(bluredSquaredDerX, bluredSquaredDerY) - \
          np.multiply(bluredDerXderY, bluredDerXderY)
    trace = bluredSquaredDerX + bluredSquaredDerY

    R = det - DEFAULT_K * (np.multiply(trace, trace))

    locMax = sol4_add.non_maximum_suppression(R)

    tmpP = np.argwhere(locMax)

    points = tmpP[:, [1, 0]]

    return points


def sample_descriptor(im, pos, desc_rad):
    '''
    find the descriptors for the points given.
    :param im: grayscale image to sample within
    :param pos:  An array with shape (N,2) of [x,y] positions to
    sample descriptors in im.
    :param desc_rad: ”Radius” of descriptors to compute (see below).
    :return: A 3D array with shape (K,K,N) containing the ith descriptor
    at desc(:,:,i). The per−descriptor dimensions KxK
    are related to the desc rad argument as follows K = 1+2∗desc rad.
    '''

    posFix = pos[:, [1, 0]]
    patchWidth = 2 * desc_rad + 1
    patchSize = patchWidth ** 2
    N = posFix.shape[0]
    patches = np.zeros((patchWidth, patchWidth, N))
    for i in range(N):
        pointI = np.repeat(posFix[i][np.newaxis], patchSize, axis=0)
        offset = np.indices((patchWidth, patchWidth))
        pointI = pointI + (np.transpose(offset.reshape(2, -1)) - desc_rad)

        patchI = map_coordinates(im, np.transpose(pointI),
                                 order=1, prefilter=False).reshape(patchWidth,
                                                                   patchWidth)
        meanI = np.mean(patchI)
        normI = np.linalg.norm(patchI)
        patchI -= meanI
        if normI == 0:
            continue

        patchI /= normI
        patches[:, :, i] = patchI
    return patches


def find_features(pyr):
    '''
    find features and descriptors.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return:
    pos − An array with shape (N,2) of [x,y] feature location per row found
    in the (third pyramid level of the) image. These coordinates are
    provided at the pyramid level pyr[0].
    desc − A feature descriptor array with shape (K,K,N).

    '''
    pos = sol4_add.spread_out_corners(pyr[ORIG_IM], DEF_N, DEF_M,
                                      DEFAULT_RADIUS)
    desc = sample_descriptor(pyr[2], pos * [0.25, 0.25], DEFAULT_DESC_RAD)
    # print(desc.shape)
    return pos, desc


def get2MaxInd(mat):
    '''
    returns a matrix of the same size as input with 2 max indices
    are 1 and other 0
    :param mat: input matrix
    :return: a matrix of the same size as input with 2 max indices
    are 1 and other 0
    '''
    out = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        m = np.argsort(mat[i, :])
        i1, i2 = m[-1], m[-2]
        out[i, i1] = 1
        out[i, i2] = 1

    return out


def match_features(desc1, desc2, min_score):
    '''
    match features in 2 images.
    :param desc1: A feature descriptor array with shape (K,K,N1)
    :param desc2: A feature descriptor array with shape (K,K,N2)
    :param min_score: Minimal match score between two descriptors
    required to be regarded as corresponding points.
    :return:
    Array with shape (M,) and dtype int of matching indices in desc1
    Array with shape (M,) and dtype int of matching indices in desc2
    '''
    K1 = desc1.shape[0]
    N1 = desc1.shape[2]
    N2 = desc2.shape[2]
    desc1Flatt = np.transpose(desc1.reshape(K1 ** 2, N1))
    desc2Flatt = desc2.reshape(K1 ** 2, N2)
    responseDescs = np.dot(desc1Flatt, desc2Flatt)
    responseDescs[responseDescs < min_score] = 0
    M1 = get2MaxInd(responseDescs)
    M2 = np.transpose(get2MaxInd(np.transpose(responseDescs)))
    matches = np.multiply(M1, M2)
    match_ind1, match_ind2 = np.where(matches == 1)

    return match_ind1, match_ind2


def apply_homography(pos1, H12):
    '''
    applies a homography transformation on a set of points
    :param pos1: − An array with shape (N,2) of [x,y] point coordinates
    :param H12: A 3x3 homography matrix
    :return: An array with the same shape as pos1 with [x,y] point
    coordinates in image i+1 obtained from transforming pos1 using H12.

    '''
    homPos1 = np.ones((pos1.shape[0], 3))
    homPos1[:, 0:2] = pos1
    homPos2 = np.dot(H12, np.transpose(homPos1))
    pos2 = np.zeros(pos1.shape)
    pos2[:, 0] = np.transpose(np.divide(homPos2[0, :], homPos2[2, :]))
    pos2[:, 1] = np.transpose(np.divide(homPos2[1, :], homPos2[2, :]))
    return pos2


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    '''
    perform Ransac
    :param pos1, pos2: Two Arrays, each with shape (N,2) containing n rows
    of [x,y]  coordinates of matched points.
    :param num_iters: Number of RANSAC iterations to perform
    :param inlier_tol: inlier tolerance threshold
    :return:
    H12 − A 3x3 normalized homography matrix
    inliers − An Array with shape (S,) where S is the number of inliers,
    containing the indices in pos1/pos2 of the maximal set of inlier
    matches found.
    '''
    N1 = pos1.shape[0]
    N2 = pos2.shape[0]
    assert N1 == N2 and num_iters >= 1 and inlier_tol > 0 and N1 >= 4

    countInliers = 0
    inliersInds = np.zeros(N1)

    for i in range(num_iters):
        curInliersInds = np.zeros(N1)
        curCountInliers = 0
        randInds = np.random.permutation(N1)[:4]
        H12 = sol4_add.least_squares_homography(pos1[randInds, :],
                                                pos2[randInds, :])
        if H12 is not None:
            homPos2 = apply_homography(pos1, H12)
            for j in range(N1):
                if np.linalg.norm(homPos2[j, :] - pos2[j, :]) ** 2 < inlier_tol:
                    curInliersInds[curCountInliers] = j
                    curCountInliers += 1
            if curCountInliers > countInliers:
                countInliers = curCountInliers
                inliersInds = curInliersInds

    inliersInds = inliersInds[inliersInds != 0].astype(np.uint64)
    H12 = sol4_add.least_squares_homography(pos1[inliersInds, :],
                                            pos2[inliersInds, :])

    # another calculation of E
    curInliersInds = np.zeros(N1)
    curCountInliers = 0
    homPos2 = apply_homography(pos1, H12)
    for j in range(N1):
        if np.linalg.norm(homPos2[j, :] - pos2[j, :]) ** 2 < inlier_tol:
            curInliersInds[curCountInliers] = j
            curCountInliers += 1

    inliersInds = curInliersInds[curInliersInds != 0].astype(np.uint64)
    return H12, inliersInds


def display_matches(im1, im2, pos1, pos2, inliers):
    '''
    visualize the full set of point matches and the inlier matches
    detected by RANSAC
    :param im1: grayscale images
    :param im2: grayscale images
    :param pos1, pos2: Two arrays with shape (N,2) each, containing N rows
    of [x,y] coordinates of matched points in im1 and im2 (i.e. the match
    of the ith coordinate is pos1[i,:] in im1 and pos2[i,:] in im2).
    :param inliers: An array with shape (S,) of inlier matches (e.g. see
    output of ransac_homography)
    '''
    stackedIm = np.hstack((im1,
                           im2))  # todo check if it ok meaning there could not be 2 images with different shapes
    newPos2 = np.zeros(pos2.shape)
    newPos2[:, 0] = pos2[:, 0] + im1.shape[COLS]
    newPos2[:, 1] = pos2[:, 1]
    plt.figure()
    plt.imshow(stackedIm, cmap=plt.cm.gray)
    plt.scatter(pos1[:, 0], pos1[:, 1], c='red', marker='.')
    plt.scatter(newPos2[:, 0], newPos2[:, 1], c='red', marker='.')

    for i in range(pos1.shape[0]):
        plt.plot([pos1[i, 0], newPos2[i, 0]], [pos1[i, 1], newPos2[i, 1]], 'b',
                 linewidth=0.3)

    print(inliers.shape)
    for j in range(inliers.shape[0]):
        plt.plot([pos1[inliers[j], 0], newPos2[inliers[j], 0]],
                 [pos1[inliers[j], 1], newPos2[inliers[j], 1]], 'y',
                 linewidth=1)
    plt.show()


def accumulate_homographies(H_successive, m):
    '''
    return new homographies from some fram to frame m
    :param H_successive: A list of M−1 3x3 homography matrices where H
    successive[i] is a homography that transforms points
    from coordinate system i to coordinate system i+1
    :param m: − Index of the coordinate system we would like to accumulate
    the given homographies towards.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms
    points from coordinate system i to coordinate system m.
    '''
    H2m = np.zeros((3, 3, len(H_successive) + 1))
    currH = np.eye(3)

    for i in range(m):
        tmp = np.dot(currH, H_successive[m - 1 - i])
        H2m[:, :, m - 1 - i] = np.divide(tmp, tmp[2, 2])
        currH = H2m[:, :, m - 1 - i]

    H2m[:, :, m] = np.eye(3)

    currH = np.eye(3)
    for i in range(len(H_successive) - m):
        tmp = np.dot(np.linalg.inv(H_successive[m + i]), currH)
        H2m[:, :, m + 1 + i] = np.divide(tmp, tmp[2, 2])
        currH = H2m[:, :, m + 1 + i]
    return [H2m[:, :, i] for i in range(H2m.shape[2])]


def render_panorama(ims, Hs):
    '''
    creates a grayscale panorama image composed of vertical strips,
    backwarped using homographies from Hs, one from every image in ims.
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography
    that transforms points from the coordinate system of ims [i] to the
    coordinate system of the panorama. (Python list)
    :return: − A grayscale panorama image composed of vertical strips,
    backwarped using homographies from Hs, one from every image in ims.
    '''

    minX, minY, maxX, maxY = getMaxAndMin(ims, Hs)
    rows = maxY - minY + 1
    cols = maxX - minX + 1
    panorama = np.zeros((rows, cols))
    xPanoVals, yPanoVals = np.meshgrid(np.linspace(minX, maxX, cols),
                                       np.linspace(minY, maxY, rows))

    stripesBound = findBounds(ims, Hs)
    stripesBound -= minX
    stripesBound = np.append([0], stripesBound)
    stripesBound = np.append(stripesBound, [cols])

    firstImGridX = xPanoVals[:, :stripesBound[2]]
    firstImGridY = yPanoVals[:, :stripesBound[2]]

    posOfStrips = np.transpose(
        np.array([firstImGridX.flatten(), firstImGridY.flatten()]))
    im1Pos = apply_homography(posOfStrips, np.linalg.inv(Hs[0]))

    firstImGridX = im1Pos[:, 0]
    firstImGridY = im1Pos[:, 1]

    panorama[:, :stripesBound[2]] = map_coordinates(ims[0], [firstImGridX,
                                                             firstImGridY],
                                                    order=1,
                                                    prefilter=False).reshape(
        panorama[:, :stripesBound[2]].shape)
    # print(panorama[:, :stripesBound[2]])

    for i in range(len(stripesBound) - 3):
        gridX = xPanoVals[:, stripesBound[i]:stripesBound[3 + i]]
        gridY = yPanoVals[:, stripesBound[i]:stripesBound[3 + i]]

        posOfStrips = np.transpose(
            np.array([gridX.flatten(), gridY.flatten()]))
        imPos = apply_homography(posOfStrips, np.linalg.inv(Hs[1 + i]))

        gridX = imPos[:, 0]
        gridY = imPos[:, 1]

        firstStripe = panorama[:, stripesBound[i]:stripesBound[i + 3]]

        secStripe = map_coordinates(ims[1 + i], [gridX, gridY], order=1,
                                    prefilter=False).reshape(
            panorama[:, stripesBound[i]:stripesBound[i + 3]].shape)

        plt.figure()
        plt.imshow(panorama, cmap=plt.cm.gray)

        panorama[:, stripesBound[i]:stripesBound[i + 3]] = stitch(firstStripe,
                                                                  secStripe)
        plt.figure()
        plt.imshow(panorama, cmap=plt.cm.gray)
        plt.show()
        print(panorama[:, stripesBound[i]:stripesBound[i + 3]])

    lastImGridX = xPanoVals[:, stripesBound[len(stripesBound) - 3]:stripesBound[
        len(stripesBound) - 1]]
    lastImGridY = yPanoVals[::,
                  stripesBound[len(stripesBound) - 3]:stripesBound[
                      len(stripesBound) - 1]]

    posOfStrips = np.transpose(
        np.array([lastImGridX.flatten(), lastImGridY.flatten()]))
    imLastPos = apply_homography(posOfStrips,
                                 np.linalg.inv(Hs[len(Hs) - 1]))

    lastImGridX = imLastPos[:, 0]
    lastImGridY = imLastPos[:, 1]

    lastStripe = map_coordinates(ims[len(ims) - 1], [lastImGridX, lastImGridY],
                                 order=1,
                                 prefilter=False).reshape(panorama[:,
                                                          stripesBound[len(
                                                              stripesBound) - 3]:
                                                          stripesBound[
                                                              len(
                                                                  stripesBound) - 1]].shape)
    oldStripe = panorama[:, stripesBound[len(stripesBound) - 3]:stripesBound[
        len(stripesBound) - 1]]

    panorama[:, stripesBound[len(stripesBound) - 3]:stripesBound[
        len(stripesBound) - 1]] = stitch(oldStripe, lastStripe)

    return panorama


def stitch(firstStripe, secStripe):
    '''
    stitch 2 strips
    :param firstStripe:
    :param secStripe:
    :return:
    '''
    PYR_LEV = 6
    firstStripe = np.nan_to_num(firstStripe)
    secStripe = np.nan_to_num(secStripe)
    rows, cols = firstStripe.shape

    overlap = np.multiply(firstStripe, secStripe)
    cumsum = np.cumsum(overlap, axis=1)
    nonZero = np.nonzero(cumsum)
    first = nonZero[1][0]
    last = nonZero[1][len(nonZero[1]) - 1]
    first += 5  # todo change if nedded
    last -= 5

    colsOverlap = last - first + 1

    # dynamic programing
    diff = np.ones((rows, colsOverlap))

    for i in range(colsOverlap):
        diff[0, i] = (firstStripe[0, first + i] - secStripe[0, first + i]) ** 2

    for j in range(rows - 1):
        # for borders todo
        diff[j + 1, 0] = (firstStripe[j + 1, first] - secStripe[j + 1,
                                                                first]) ** 2 + np.min(
            diff[j, :2])
        diff[j + 1, colsOverlap - 1] = (firstStripe[j + 1, last] - secStripe[
            j + 1,
            last]) ** 2 + np.min(diff[j, colsOverlap - 2:])

        for i in range(colsOverlap - 2):
            diff[j + 1, i + 1] = (firstStripe[j + 1, first + i + 1] - secStripe[
                j + 1,
                first + i + 1]) ** 2 + np.min(diff[j, i:i + 3])
    stitchArr = np.zeros((1, rows))
    ind = np.argmin(diff[rows - 1, 1:colsOverlap - 1])
    stitchArr[0, rows - 1] = ind + 1

    for i in range(rows - 1):
        l = int((stitchArr[0, rows - 1 - i] - 1))
        r = int((stitchArr[0, rows - 1 - i] + 1))
        ind = np.argmin(diff[rows - 1 - i, l:r + 1])
        if ind == 0:
            ind = 1
        if ind == colsOverlap - 1:
            ind = colsOverlap - 2
        stitchArr[0, rows - 1 - i - 1] = ind + l - 1

    stitchArr = stitchArr + first - 1

    # padd for blending
    tmpR = int(np.ceil(rows / (2 ** PYR_LEV)) * 2 ** PYR_LEV)
    tmpC = int(np.ceil(cols / (2 ** PYR_LEV)) * 2 ** PYR_LEV)
    tmpS = np.zeros((tmpR, tmpC))
    tmpS[:rows, :cols] = firstStripe
    tmpS1 = tmpS
    tmpS[:rows, :cols] = secStripe
    tmpS2 = tmpS

    mask = np.zeros(tmpS1.shape)
    for i in range(rows):
        mask[i, :int((stitchArr[0, i]))] = 1

    res = sol4_utils.pyramid_blending(tmpS1, tmpS2, mask, PYR_LEV, 5, 5)
    print(res)
    return res[:rows, :cols]


def findBounds(ims, Hs):
    '''
    find the strips x bounds
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography
    that transforms points from the coordinate system of ims [i] to the
    coordinate system of the panorama. (Python list)
    :param ims: A list of grayscale images. (Python list)
    :return: the strips x bounds
    '''

    bounds = np.zeros(len(ims) - 1).astype(np.int64)

    for i in range(len(ims) - 1):
        xIcenter = np.ceil(ims[i].shape[COLS] / 2)
        yIcenter = np.ceil(ims[i].shape[ROWS] / 2)
        xIp1center = np.ceil(ims[i + 1].shape[COLS] / 2)
        yIp1center = np.ceil(ims[i].shape[ROWS] / 2)

        ihomCenter = apply_homography(
            np.array([xIcenter, yIcenter]).reshape(1, 2), Hs[i])
        ip1homCenter = apply_homography(
            np.array([xIp1center, yIp1center]).reshape(1, 2), Hs[i + 1])

        bounds[i] = np.ceil((ihomCenter[0][ROWS] + ip1homCenter[0][ROWS]) / 2)

    return bounds


def getMaxAndMin(ims, Hs):
    '''
    find min and max x and y for panorama
    :param ims: A list of grayscale images. (Python list)
    :param Hs:  A list of 3x3 homography matrices. Hs[i] is a homography
    that transforms points from the coordinate system of ims [i] to the
    coordinate system of the panorama. (Python list)
    :return: min and max x and y for panorama
    '''

    numIm = len(ims)
    xAndy = np.zeros((4 * numIm, 2))
    for i in range(numIm):
        numRows, numCols = ims[i].shape
        edges = [[0, 0], [numCols - 1, 0], [0, numRows - 1],
                 [numCols - 1, numRows - 1]]
        edges = np.array(edges)
        homEdges = apply_homography(edges, Hs[i])
        # print(homEdges)
        xAndy[4 * i] = homEdges[0]
        xAndy[4 * i + 1] = homEdges[1]
        xAndy[4 * i + 2] = homEdges[2]
        xAndy[4 * i + 3] = homEdges[3]

    return [np.floor(np.min(xAndy[:, 0])).astype(np.int64),
            np.floor(np.min(xAndy[:, 1])).astype(np.int64),
            np.ceil(np.max(xAndy[:, 0])).astype(np.int64),
            np.ceil(np.max(xAndy[:, 1])).astype(np.int64)]
