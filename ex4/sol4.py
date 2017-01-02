import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
from scipy.ndimage.filters import convolve
import os

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


def getRespons(M, imShape):
    print(imShape)
    print(M.shape)
    for row in imShape[ROWS]:
        for col in imShape[COLS]:
            mPerPoint = [M[0][ROWS][COLS], M[1][ROWS][COLS],
                         M[2][ROWS][COLS], M[3][ROWS][COLS]]
            print(mPerPoint.shape)


    eigenValues, eigenVectores = np.linalg.eig(M)
    print(eigenValues)
    pass

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


    M = np.array([[bluredSquaredDerX, bluredDerXderY],
                  [bluredDerXderY, bluredSquaredDerY]])

    respone = getRespons(M, im.shape)
