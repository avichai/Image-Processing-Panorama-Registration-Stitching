import matplotlib.pyplot as plt
import numpy as np
import os

import time as time  # todo remove
from ex4 import sol4  # todo remove from ex4
from ex4 import sol4_utils  # todo remove from ex4


def generate_panorama(data_dir, file_prefix, num_images, figsize=(20, 20)):
    t = time.time()
    # The naming convention for a sequence of images is nameN.jpg, where N is a running number 1,2,..
    files = [os.path.join(data_dir, '%s%d.jpg' % (file_prefix, i + 1)) for i in
             range(num_images)]

    # Read images.
    ims = [sol4_utils.read_image(f, 1) for f in files]

    # Extract feature point locations and descriptors.
    def im_to_points(im):
        pyr, _ = sol4_utils.build_gaussian_pyramid(im, 3, 7)
        return sol4.find_features(pyr)

    p_d = [im_to_points(im) for im in ims]

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(num_images - 1):
        points1, points2 = p_d[i][0], p_d[i + 1][0]
        desc1, desc2 = p_d[i][1], p_d[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = sol4.match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = sol4.ransac_homography(points1, points2, 10000, 6)

        # Display inlier and outlier matches.
        sol4.display_matches(ims[i], ims[i + 1], points1, points2,
                             inliers=inliers)
        Hs.append(H12)

    # Compute composite homographies from the panorama coordinate system.
    Htot = sol4.accumulate_homographies(Hs, (num_images - 1) // 2)

    # Final panorama is generated using 3 channels of the RGB images
    ims_rgb = [sol4_utils.read_image(f, 2) for f in files]

    # Render panorama for each color channel and combine them.
    panorama = [sol4.render_panorama([im[..., i] for im in ims_rgb], Htot) for i
                in range(3)]
    panorama = np.dstack(panorama)

    # plot the panorama
    plt.figure(figsize=figsize)
    plt.imshow(panorama.clip(0, 1))
    print(time.time()-t)
    plt.show()


def main():
    generate_panorama('external/', 'office', 4)
    # generate_panorama('external/', 'backyard', 3, (20,10))
    # generate_panorama('external/', 'oxford'  , 2)


if __name__ == '__main__':
    main()
