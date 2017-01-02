from ex4 import sol4 as sol4

import matplotlib.pyplot as plt
import numpy as np


def testHarris():
    im = sol4.read_image('external/backyard1.jpg', 1)
    # plt.figure()
    # plt.imshow(im, cmap=plt.cm.gray)
    # plt.show()
    pos = sol4.harris_corner_detector(im)





#tests
# testHarris


def main():
    try:
        for test in [testHarris]:
            test()
    except Exception as e:
        print('Failed test due to: {0}'.format(e))
        exit(-1)

if __name__ == '__main__':
    main()