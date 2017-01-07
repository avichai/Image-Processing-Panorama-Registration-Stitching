import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates
import os
import itertools

from ex4 import sol4_add

IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]
GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
ROWS = 0
COLS = 1
LARGEST_IM_INDEX = 0
DIM_RGB = 3
DER_VEC = [1, 0, -1]
DEFAULT_KER_SIZE = 3
DEFAULT_K = 0.4
FIRST_EIG_VAL_IND = 0
SEC_EIG_VAL_IND = 0
ORIG_IM = 0
DEFAULT_DESC_RAD = 5
DEF_N = 7
DEF_M = 11


def read_image(filename, representation):
    """this function reads a given image file and converts it into a given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        if np.max(im) > 1:
            '''not suppose to happened'''
            im /= NORM_PIX_FACTOR
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def getGaussVec(kernel_size):
    '''
    gets the gaussian vector in the length of the kernel size
    :param kernel_size: the length of the wished kernel
    :return: the 1d vector we want
    '''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return [1]
    return sig.convolve(BINOMIAL_MAT, getGaussVec(kernel_size - 1)).astype(
        np.float32)


def getImAfterBlur(im, filter):
    '''
    return the image after row and col blur
    :param im: the image to blur
    :param filter: the filter to blur with
    :return: blurred image
    '''
    blurXIm = convolve(im, filter)
    blurIm = convolve(blurXIm, filter.transpose())
    return blurIm


def reduceIm(currIm, gaussFilter, filter_size):
    '''
    reduce an image
    :param currIm: the image to reduce by 4
    :param gaussFilter: the filter to blur with the image before reduce
    :param filter_size: the size of the filter
    :return: the reduced image
    '''
    blurIm = getImAfterBlur(currIm, gaussFilter)
    reducedImage = blurIm[::2, ::2]
    return reducedImage.astype(np.float32)


def expandIm(currIm, gaussFilterForExpand, filter_size):
    '''
    expand an image
    :rtype : np.float32
    :param currIm: the image to expand by 4
    :param gaussFilterForExpand: the filter to blur with the expand image
    :param filter_size: the size of the filter
    :return: an expand image
    '''
    expandImage = np.zeros((2 * currIm.shape[0], 2 * currIm.shape[1]))
    expandImage[::2, ::2] = currIm
    expandRes = getImAfterBlur(expandImage, gaussFilterForExpand)
    return expandRes.astype(np.float32)


def getNumInInPyr(im, max_levels):
    '''
    return maximum number of images in pyramid
    :param im: tne original image
    :param max_levels: an initial limitation
    :return: the real limitation
    '''
    numRows, numCols = im.shape[ROWS], im.shape[COLS]

    limRows = np.floor(np.log2(numRows)) - 3
    limCols = np.floor(np.log2(numCols)) - 3
    numImInPyr = np.uint8(np.min([max_levels, limCols, limRows]))
    return numImInPyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Gaussian pyramid as standard python array and the filter vec
    '''
    numImInPyr = getNumInInPyr(im, max_levels)
    gaussFilter = np.array(getGaussVec(filter_size)).reshape(1, filter_size)

    gaussPyr = [im]
    currIm = im
    for i in range(1, numImInPyr):
        currIm = reduceIm(currIm, gaussFilter, filter_size)
        gaussPyr.append(currIm)
    return gaussPyr, gaussFilter


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Laplacian pyramid as standard python array and the filter vec
    '''
    gaussFilter = np.array(getGaussVec(filter_size)).reshape(1, filter_size)
    laplacianPyr = []

    gaussPyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    numImInPyr = len(gaussPyr)

    for i in range(numImInPyr - 1):
        laplacianPyr.append(gaussPyr[i] - expandIm(
            gaussPyr[i + 1], np.multiply(2, gaussFilter), filter_size))
    laplacianPyr.append(gaussPyr[numImInPyr - 1])
    return laplacianPyr, gaussFilter


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    reconstruction of an image from its Laplacian Pyramid
    :param lpyr: Laplacian pyramid
    :param filter_vec: the filter that was used in order to
            construct the pyramid
    :param coeff: the coefficient of each image in the pyramid
    :return: reconstruction of an image from its Laplacian Pyramid
    '''
    numIm, numCoe = len(lpyr), len(coeff)
    if numIm != numCoe:
        '''invalid input'''
        return
    gni = lpyr[numIm - 1]
    for i in range(numIm - 1):
        gni = expandIm(gni, np.multiply(2, filter_vec), len(filter_vec)) + (
            lpyr[numIm - 1 - i - 1] * coeff[len(coeff) - 1 - i])
    return gni.astype(np.float32)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    '''
    blending images using pyramids.
    :rtype : tuple
    :param im1: first im to blend - grayscale
    :param im2: second im to blend - grayscale
    :param mask:  is a boolean (i.e. dtype == np.bool) mask containing
        True and False representing which parts of im1 and im2 should
        appear in the resulting im_blend.
    :param max_levels:  is the max_levels parameter you should use when
            generating the Gaussian and Laplacian pyramids
    :param filter_size_im:  is the size of the Gaussian filter
            (an odd scalar that represents a squared filter) which defining the
            filter used in the construction of the Laplacian pyramids of
            im1 and im2
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask.
    :return: blended image using laplacian and gausian pyramids.
    '''
    l1Pyr, filterVec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2Pyr, filterVec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    maskInFlots = mask.astype(np.float32)
    gaussMaskPyr, filterVec3 = build_gaussian_pyramid(maskInFlots, max_levels,
                                                      filter_size_mask)
    lOut = []
    lenOfPyr = len(l1Pyr)
    for i in range(lenOfPyr):
        lOut.append(np.multiply(gaussMaskPyr[i], l1Pyr[i]) +
                    (np.multiply(1 - gaussMaskPyr[i], l2Pyr[i])))
    blendedIm = laplacian_to_image(lOut, filterVec1, [1] * lenOfPyr)
    blendedImClip = np.clip(blendedIm, 0, 1)
    return blendedImClip.astype(np.float32)


def getBlurMat(kernel_size):
    '''
    getting a blur kernel of size kernel_sise^2
    :param kernel_size: the size of the wished kernel in
    each dimension (an odd integer)
    :return: blur kernel of size kernel_sise^2
    '''
    '''geeting the blure vec in 1d'''
    blurVec = getGaussVec(kernel_size)
    '''creating the 2d kernel'''
    blurAsMat = np.array(blurVec)
    blurMat = sig.convolve2d(blurAsMat.reshape(kernel_size, 1),
                             blurAsMat.reshape(1, kernel_size))
    return blurMat


def blur_spatial(im, kernel_size):
    '''
    function that performs image blurring using 2D convolution
    between the image f and a gaussian kernel g.
    :param im: image to be blurred (grayscale float32 image).
    :param kernel_size: is the size of the gaussian kernel in
    each dimension (an odd integer)
    :return:  the output blurry image (grayscale float32 image).
    '''
    '''the kernel will do nothing'''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return im
    '''getting the bluring matrix'''
    blurMat = getBlurMat(kernel_size)
    return sig.convolve2d(im, blurMat, mode='same', boundary="wrap").astype(
        np.float32)


def conv_der(im):
    '''
    getting the derivative of an image using convolution
    :param im:  grayscale images of type float32.
    :return:  X and Y derivative of an image.
    ima.
    '''
    maskX = np.array(DER_VEC, ndmin=2)
    maskY = np.transpose(maskX)
    derX = sig.convolve(im, maskX, mode='same')
    derY = sig.convolve(im, maskY, mode='same')
    return derX.astype(np.float32), derY.astype(np.float32)


def harris_corner_detector(im):
    '''
    Finding haris point of interest
    :param im: grayscale image to find key points inside
    :return: An array with shape (N,2) of [x,y] key points locations in im
    '''
    derX, derY = conv_der(im)

    bluredSquaredDerX = blur_spatial(np.multiply(derX, derX), DEFAULT_KER_SIZE)
    bluredSquaredDerY = blur_spatial(np.multiply(derY, derY), DEFAULT_KER_SIZE)
    bluredDerXderY = blur_spatial(np.multiply(derX, derY), DEFAULT_KER_SIZE)

    R = np.multiply(bluredSquaredDerX, bluredSquaredDerY) - \
        np.multiply(bluredDerXderY, bluredDerXderY)

    locMax = sol4_add.non_maximum_suppression(R)
    y, x = np.where(locMax == True)

    pos = np.zeros((x.shape[0], 2))
    pos[:, 0] = x
    pos[:, 1] = y

    return pos


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

    patchSize = 2 * desc_rad + 1
    N = pos.shape[0]
    patches = np.zeros((patchSize, patchSize, N))
    xCoord = np.zeros(patchSize ** 2)
    yCoord = np.zeros(patchSize ** 2)
    c = 0
    for i in range(N):
        pljX = 0.25 * pos[i][ROWS]
        pljY = 0.25 * pos[i][COLS]
        for j in range(patchSize):
            for k in range(patchSize):
                xCoord[j * patchSize + k] = pljX - desc_rad + j
                yCoord[j + k * patchSize] = pljY - desc_rad + j
        patchI = map_coordinates(im, [xCoord, yCoord],
                                 order=1, prefilter=False).reshape(patchSize,
                                                                   patchSize)
        meanI = np.mean(patchI)
        patchI -= meanI
        normI = np.linalg.norm(patchI)
        if normI == 0:
            patches[:, :, i] = np.zeros((patchSize, patchSize))
            continue

        patchI /= np.linalg.norm(patchI)
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
                                      DEFAULT_DESC_RAD)
    desc = sample_descriptor(pyr[2], pos, DEFAULT_DESC_RAD)
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
    K2 = desc2.shape[0]
    assert K1 == K2  # todo maybe remove
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
    assert N1 == N2 and num_iters >= 1 and inlier_tol > 0

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

    inliersInds = inliersInds[inliersInds != 0].astype(np.uint8)
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

    inliersInds = curInliersInds[curInliersInds != 0].astype(np.uint8)
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
    stackedIm = np.hstack((im1, im2)) # todo check if it ok meaning there could not be 2 images with different shapes
    newPos2 = np.zeros(pos2.shape)
    newPos2[:, 0] = pos2[:, 0] + im1.shape[COLS]
    newPos2[:, 1] = pos2[:, 1]
    plt.figure()
    plt.imshow(stackedIm, cmap=plt.cm.gray)
    plt.scatter(pos1[:, 0], pos1[:, 1], c='red', marker='.')
    plt.scatter(newPos2[:, 0], newPos2[:, 1], c='red', marker='.')

    for i in range(pos1.shape[0]):
        plt.plot([pos1[i, 0], newPos2[i, 0]], [pos1[i, 1], newPos2[i, 1]], 'b', linewidth=0.3)

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
    H2m = np.zeros((3, 3, len(H_successive)))
    currH = np.eye(3)

    for i in range(m):
        tmp = np.dot(currH, H_successive[m-1-i])
        H2m[:, :, m-1-i] = np.divide(tmp, tmp[2, 2])
        currH = H2m[:, :, m-1-i]

    H2m[:, :, m] = np.eye(3)

    currH = np.eye(3)
    for i in range(len(H_successive) - m - 1):
        tmp = np.dot(np.linalg.inv(H_successive[m+i]), currH)
        H2m[:, :, m+1+i] = np.divide(tmp, tmp[2, 2])
        currH = H2m[:, :, m+1+i]
    return H2m
