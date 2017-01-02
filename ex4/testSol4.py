from ex4 import sol4 as sol4

import matplotlib.pyplot as plt
import numpy as np


def testHarris(im):

    pos = sol4.harris_corner_detector(im)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(pos[:, 1], pos[:, 0], marker='.')
    plt.show()

def testSampleDesc(im):
    pyr_im, filter = sol4.build_gaussian_pyramid(im, 3, 3)
    pos = sol4.harris_corner_detector(im)
    desc_rad = 3
    desc = sol4.sample_descriptor(pyr_im[2], pos, desc_rad)

    from scipy import ndimage
    a = np.arange(12.).reshape((4, 3))
    print(a)

    print(ndimage.map_coordinates(a, [[2.5, 2], [0.5, 1]], order=1))

def testFindFeatures(im):
    pyr = sol4.build_gaussian_pyramid(im, 3, 3)
    pos, desc = sol4.find_features(pyr)










#tests
# testHarris
# testSampleDesc
# testFindFeatures


def main():
    try:
        im = sol4.read_image('external/backyard1.jpg', 1)
        for test in [testSampleDesc]:
            test(im)
    except Exception as e:
        print('Failed test due to: {0}'.format(e))
        exit(-1)

if __name__ == '__main__':
    main()