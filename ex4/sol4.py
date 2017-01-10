import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates

from ex4 import sol4_add
from ex4 import sol4_utils

ROWS = 0
COLS = 1
DER_VEC = np.array([1, 0, -1])[np.newaxis]
DEFAULT_KER_SIZE = 3
DEFAULT_K = 0.04
FIRST_EIG_VAL_IND = 0
SEC_EIG_VAL_IND = 0
ORIG_IM = 0
DEFAULT_DESC_RAD = 3
DEFAULT_RADIUS = 12
DEF_N = 3
DEF_M = 3
ITER_FIRST_IMAGE = 0
EXPAND_FOR_BLEND_FACTOR = 0.4
PYR_LEVEL = 4
DEF_FILTER_IM = 3
DEF_FILTER_MASK = 3


def harris_corner_detector(im):
    '''
    Finding haris point of interest
    :param im: grayscale image to find key points inside
    :return: An array with shape (N,2) of [x,y] key points locations in im
    '''
    derX = convolve(im, DER_VEC)
    derY = convolve(im, np.transpose(DER_VEC))

    bluredSquaredDerX = sol4_utils.blur_spatial(np.multiply(derX, derX),
                                                DEFAULT_KER_SIZE)
    bluredSquaredDerY = sol4_utils.blur_spatial(np.multiply(derY, derY),
                                                DEFAULT_KER_SIZE)
    bluredDerXderY = sol4_utils.blur_spatial(np.multiply(derX, derY),
                                             DEFAULT_KER_SIZE)

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
        patchI -= meanI
        normI = np.linalg.norm(patchI)
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
    # print(H12)
    homPos1 = np.ones((pos1.shape[0], 3))
    homPos1[:, 0:2] = pos1
    # print(homPos1)
    homPos2 = np.dot(H12, np.transpose(homPos1))
    homPos2[2, :][homPos2[2, :] == 0] += 1e-50
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
        randInds = np.random.permutation(N1)[:4]
        H12 = sol4_add.least_squares_homography(pos1[randInds, :],
                                                pos2[randInds, :])

        countInliers, inliersInds = updateInliers(H12, pos1, pos2, inlier_tol,
                                                  countInliers, inliersInds)

    inliersInds = np.nonzero(inliersInds)[0]
    H12 = sol4_add.least_squares_homography(pos1[inliersInds, :],
                                            pos2[inliersInds, :])
    return H12, inliersInds


def updateInliers(H12, pos1, pos2, inlier_tol,
                  countInliers, inliersInds):
    '''
    update counInliers and inliersInds
    :param H12:
    :param pos1:
    :param pos2:
    :param inlier_tol:
    :param countInliers:
    :param inliersInds:
    :return: updated counInliers and inliersInds
    '''
    if H12 is not None:
        homPos2 = apply_homography(pos1, H12)

        curInliersInds = np.sum(np.square(homPos2 - pos2), axis=1) < inlier_tol
        curCountInliers = np.sum(curInliersInds)
        if curCountInliers > countInliers:
            countInliers = curCountInliers
            inliersInds = curInliersInds

    return countInliers, inliersInds


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
    stackedIm = np.hstack((im1, im2))
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

    # print(inliers.shape)
    for j in range(inliers.shape[0]):
        plt.plot([pos1[inliers[j], 0], newPos2[inliers[j], 0]],
                 [pos1[inliers[j], 1], newPos2[inliers[j], 1]], 'y',
                 linewidth=1)


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
        tmp = np.dot(currH, np.linalg.inv(H_successive[m + i]))
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
    minX, minY, maxX, maxY, edges, homEdges = getMaxAndMin(ims, Hs)
    rows = maxY - minY
    cols = maxX - minX
    panorama = np.zeros((rows, cols))
    stripesBound = findBounds(ims, homEdges)

    for i in range(len(ims)):
        xLeft, xRight = stripesBound[i]
        xs, ys = np.meshgrid(range(xLeft, xRight), range(minY, maxY))
        panStripe = np.transpose(np.vstack((np.hstack(xs), np.hstack(ys))))
        imStripe = apply_homography(panStripe, np.linalg.inv(Hs[i]))
        stripVec = map_coordinates(ims[i], np.transpose(imStripe[:, [1, 0]]),
                                   order=1, prefilter=False)
        imStripe = stripVec.reshape((rows, xRight - xLeft))

        xLeft -= minX
        xRight -= minX

        if i == ITER_FIRST_IMAGE:
            panorama[:, xLeft:xRight] = imStripe
            continue
        dupPanForBlend = np.zeros(panorama.shape)
        dupPanForBlend[:, xLeft:xRight] = imStripe
        newL = xLeft
        newR = stripesBound[i - 1][1] - minX

        ofset = int((newR - newL) * EXPAND_FOR_BLEND_FACTOR)
        newL += ofset
        newR -= ofset

        error = np.square(dupPanForBlend[:, newL:newR] -
                          panorama[:, newL:newR])

        comulativeError = np.zeros(error.shape)
        root = np.zeros(error.shape, dtype=np.int64)
        path = np.zeros(error.shape[ROWS], dtype=np.int64)

        comulativeError[0, :] = error[0, :]
        for j in range(1, error.shape[0]):
            prevErr = np.vstack(
                (np.insert(comulativeError[j - 1, :-1], 0, 1e50),
                 comulativeError[j - 1, :],
                 np.append(comulativeError[j - 1, 1:], 1e50)))
            currErr = np.min(prevErr, axis=0)
            currArgErr = np.argmin(prevErr, axis=0)

            comulativeError[j, :] = error[j, :] + currErr
            root[j, :] = np.arange(error.shape[COLS]) + currArgErr - 1

        path[-1] = currPointer = np.argmin(comulativeError[-1, :])

        for j in range(1, error.shape[0]):
            path[-1 - i] = currPointer = root[-i, currPointer]

        path = path + newL

        mask = np.transpose(np.transpose(np.indices(panorama.shape)[1]) < path)
        panorama = sol4_utils.pyramid_blending(panorama, dupPanForBlend, mask,
                                               PYR_LEVEL, DEF_FILTER_IM,
                                               DEF_FILTER_MASK)

    return panorama.astype(np.float32)


def findBounds(ims, homEdges):
    '''
    find the strips x bounds
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography
    that transforms points from the coordinate system of ims [i] to the
    coordinate system of the panorama. (Python list)
    :param ims: A list of grayscale images. (Python list)
    :param edges: im edges
    :param homEdges: hom edges
    :return: the strips x bounds
    '''

    centers = []
    for i in range(len(ims)):
        rows, cols = ims[i].shape
        centers.append(np.array([rows // 2, cols // 2]))

    xEdges = homEdges[:, 0]

    tmpXedges = xEdges.reshape(len(ims), 4)
    return np.transpose(np.vstack((np.min(tmpXedges[:, [0, 1]], axis=1),
                                   np.max(tmpXedges[:, [2, 3]],
                                          axis=1)))).astype(np.int64)


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
    edges = None
    for i in range(numIm):
        numRows, numCols = ims[i].shape
        edges = [[0, 0], [0, numRows - 1], [numCols - 1, 0],
                 [numCols - 1, numRows - 1]]
        edges = np.array(edges)
        homEdges = apply_homography(edges, Hs[i])
        # print(homEdges)
        xAndy[4 * i] = homEdges[0]
        xAndy[4 * i + 1] = homEdges[1]
        xAndy[4 * i + 2] = homEdges[2]
        xAndy[4 * i + 3] = homEdges[3]

    minX = np.floor(np.min(xAndy[:, 0])).astype(np.int64)
    minY = np.floor(np.min(xAndy[:, 1])).astype(np.int64)
    maxX = np.ceil(np.max(xAndy[:, 0])).astype(np.int64)
    maxY = np.ceil(np.max(xAndy[:, 1])).astype(np.int64)

    fixFactor = 2 ** (PYR_LEVEL - 1)
    fixForPyrX = (maxX - minX) % fixFactor
    fixForPyrY = (maxY - minY) % fixFactor
    if fixForPyrX != 0:
        maxX += fixFactor - fixForPyrX
    if fixForPyrY != 0:
        maxY += fixFactor - fixForPyrY

    return [minX, minY, maxX, maxY, edges, xAndy]
