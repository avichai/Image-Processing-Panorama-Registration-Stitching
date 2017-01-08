from ex4 import sol4 as sol4
from ex4 import sol4_utils

import matplotlib.pyplot as plt
import numpy as np

INLIER_TOL = 6

DEF_NUM_ITER = 1000

DEF_MIN_SCORE = 0.0

def testHarris(im):

    pos = sol4.harris_corner_detector(im)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(pos[:, 0], pos[:, 1], marker='.')
    plt.show()

def testSampleDesc(im):
    pyr_im, filter = sol4.build_gaussian_pyramid(im, 3, 3)
    pos = sol4.harris_corner_detector(im)
    desc_rad = 3
    desc = sol4.sample_descriptor(pyr_im[2], pos, desc_rad)

    # todo add a test for descriptor

    # from scipy import ndimage
    # a = np.arange(12.).reshape((4, 3))
    # print(a)
    #
    # print(ndimage.map_coordinates(a, [[2.5, 2], [0.5, 1]], order=1))

def testFindFeatures(im):
    pyr, filter = sol4.build_gaussian_pyramid(im, 3, 3)
    pos, desc = sol4.find_features(pyr)

    # todo add a test for descriptor

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(pos[:, 0], pos[:, 1], marker='.')
    plt.show()


def randomTest(im):
    x = np.array([1,2,3])
    y = np.array([4,5,6, 7, 8])
    xv , yv = np.meshgrid(x, y)
    print(xv)
    print(yv)

    t = np.linspace(12, 22, 11)
    print(t)
    # a = np.zeros((3, 3), dtype=bool)
    # a[0, :] = np.array([True, False, True])
    # a[1, :] = np.array([True, False, True])
    # a[2, :] = np.array([True, False, True])
    # row, col = np.where(a == True)
    #
    #
    # a = np.zeros((3, 3))
    # a[0, :] = np.array([20, 2, 3])
    # a[1, :] = np.array([10, 5, 4])
    # a[2, :] = np.array([7, 8, 9])
    #
    # out = sol4.get2MaxInd(a)
    # print(out)
    #
    # match_ind1, match_ind2 = np.where(out == 1)
    # print(match_ind1, match_ind2)



def testMatchFeatures(im1):

    # todo this is the same picture try different ones
    # im2 = sol4.read_image('external/backyard1.jpg', 1)
    im2 = sol4_utils.read_image('external/office2.jpg', 1)


    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos2, desc2 = sol4_utils.find_features(pyr2)
    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    # print(match_ind2[match_ind1 != match_ind2])

    plt.figure()
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(pos1[:, 0], pos1[:, 1], marker='.')
    plt.figure()
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(pos1[match_ind1, 0], pos1[match_ind1, 1], marker='.')
    plt.figure()
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(pos2[:, 0], pos2[:, 1], marker='.')
    plt.figure()
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(pos2[match_ind2, 0], pos2[match_ind2, 1], marker='.')
    plt.show()


def testAppHom(im):
    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)

    H12 = np.diag([1, 2, 3])
    pos2 = sol4.apply_homography(pos1, H12)
    print(pos2[0])


def testRansac(im1):


    im2 = sol4_utils.read_image('external/office2.jpg', 1)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H12, inliers = sol4.ransac_homography(pos1[match_ind1, :],
                                          pos2[match_ind2, :],
                                          DEF_NUM_ITER, INLIER_TOL)
    print(H12)

def display_matches(im1):
    # im2 = sol4.read_image('external/backyard1.jpg', 1)

    im2 = sol4_utils.read_image('external/office2.jpg', 1)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H12, inliers = sol4.ransac_homography(pos1[match_ind1, :],
                                          pos2[match_ind2, :],
                                          DEF_NUM_ITER, INLIER_TOL)

    sol4.display_matches(im1, im2, pos1[match_ind1, :], pos2[match_ind2, :], inliers)


def testAccHom(im1):
    im2 = sol4_utils.read_image('external/office2.jpg', 1)
    im3 = sol4_utils.read_image('external/office3.jpg', 1)
    im4 = sol4_utils.read_image('external/office4.jpg', 1)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H12, inliers1 = sol4.ransac_homography(pos1[match_ind1, :],
                                          pos2[match_ind2, :],
                                          DEF_NUM_ITER, INLIER_TOL)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im3, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H23, inliers2 = sol4.ransac_homography(pos1[match_ind1, :],
                                          pos2[match_ind2, :],
                                          DEF_NUM_ITER, INLIER_TOL)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im3, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im4, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H34, inliers3 = sol4.ransac_homography(pos1[match_ind1, :],
                                          pos2[match_ind2, :],
                                          DEF_NUM_ITER, INLIER_TOL)

    H_successive = [H12, H23, H34]

    m = (len(H_successive)-1)//2

    H2m = sol4.accumulate_homographies(H_successive, m)
    # tmp = sol4.accumulate_homographies1(H_successive, m)
    print(H2m[2])
    # print(tmp[2])


def testRenderPan(im1):
    im2 = sol4_utils.read_image('external/office2.jpg', 1)
    im3 = sol4_utils.read_image('external/office3.jpg', 1)
    im4 = sol4_utils.read_image('external/office4.jpg', 1)

    im1 = cutImages(im1)
    im2 = cutImages(im2)
    im3 = cutImages(im3)
    im4 = cutImages(im4)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H12, inliers1 = sol4.ransac_homography(pos1[match_ind1, :],
                                           pos2[match_ind2, :],
                                           DEF_NUM_ITER, INLIER_TOL)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im3, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H23, inliers2 = sol4.ransac_homography(pos1[match_ind1, :],
                                           pos2[match_ind2, :],
                                           DEF_NUM_ITER, INLIER_TOL)

    pyr1, filter1 = sol4_utils.build_gaussian_pyramid(im3, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    pyr2, filter2 = sol4_utils.build_gaussian_pyramid(im4, 3, 3)
    pos2, desc2 = sol4.find_features(pyr2)

    match_ind1, match_ind2 = sol4.match_features(desc1, desc2, DEF_MIN_SCORE)

    H34, inliers3 = sol4.ransac_homography(pos1[match_ind1, :],
                                           pos2[match_ind2, :],
                                           DEF_NUM_ITER, INLIER_TOL)

    H_successive = [H12, H23, H34]

    m = (len(H_successive)-1)//2

    H2m = sol4.accumulate_homographies(H_successive, m)
    ims = [im1, im2, im3, im4]
    panorama = sol4.render_panorama(ims, H2m)


def cutImages(im):
    '''
    cutting images to fit dimensions of power of 2.
    :param firstStripe:
    :param secStripe:
    :param mask:
    :return:
    '''
    cutIm1 = im[
             :2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
             :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    return cutIm1




# randomTest


# tests
# testHarris
# testSampleDesc
# testFindFeatures
# testMatchFeatures
# display_matches
# testAppHom
# testRansac
# testAccHom
# testRenderPan


def main():
    try:
        # im = sol4_utils.read_image('external/backyard1.jpg', 1)
        im = sol4_utils.read_image('external/office1.jpg', 1)
        for test in [testAccHom]:
            test(im)
    except Exception as e:
        print('Failed test due to: {0}'.format(e))
        exit(-1)

if __name__ == '__main__':
    main()