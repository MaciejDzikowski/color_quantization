"""
color quantization using k-means clustering and floyd-steinberg dithering
"""

import argparse
import numpy as np
import copy
from PIL import Image
from sklearn import cluster

# using argparse
parser = argparse.ArgumentParser(description='Color quantization')
parser.add_argument('-f', '--file', help='input rgb image', required=True)
parser.add_argument('-n', '--num_colors', type=int, help='number of colors',
                    required=False)
args = parser.parse_args()


def c_q(pic, n_colors=16):
    """
    opening, converting image to array and starting process of quantization,
    then saving output
    """
    try:
        im = np.array(Image.open(pic))[...,:3]
    except:
        raise Exception('Cannot load the image!')

    # changing the array into 2 dimensional array
    x, y, z = im.shape
    im_2d = im.reshape(x * y, z)

    # k-means clustering
    km_clstr = cluster.KMeans(n_clusters=n_colors)
    km_clstr.fit(im_2d)
    cluster_labels = km_clstr.labels_
    cluster_centers = km_clstr.cluster_centers_

    # creating 2d numpy array
    clus = cluster_centers[cluster_labels]

    # creating a list of colors in the array
    colors = [x[0] * 65536 + x[1] * 255 + x[0] for x in cluster_centers]

    # reconverting a new array to 3 dimensional
    new = clus.reshape(x, y, z)
    new2 = copy.deepcopy(new)
    # Floyd-Steinberg Dithering
    for y in range(len(new) - 1):
        for x in range(len(new[y]) - 1):
            # old values of pixel
            oldr = im[y][x][0]
            oldg = im[y][x][1]
            oldb = im[y][x][2]

            # getting quantization errors' values
            err_r = oldr - new[y][x][0]
            err_g = oldg - new[y][x][1]
            err_b = oldb - new[y][x][2]

            # changing neighbouring pixels according to algorithm
            # right neighbour
            test_r1 = new[y][x + 1][0] + err_r * 7 / 16.0
            test_g1 = new[y][x + 1][1] + err_g * 7 / 16.0
            test_b1 = new[y][x + 1][2] + err_b * 7 / 16.0
            test_pixel1 = test_r1 * 65536 + test_g1 * 256 + test_b1
            pixel1 = cluster_centers[colors.index(min(colors,
                                                     key=lambda col:
                                                     abs(col - test_pixel1)))]
            new2[y][x + 1][0] = pixel1[0]
            new2[y][x + 1][1] = pixel1[1]
            new2[y][x + 1][2] = pixel1[2]

            # bottom left-hand corner neighbour
            test_r2 = new[y][x + 1][0] + err_r * 3 / 16.0
            test_g2 = new[y][x + 1][1] + err_g * 3 / 16.0
            test_b2 = new[y][x + 1][2] + err_b * 3 / 16.0
            test_pixel2 = test_r2 * 65536 + test_g2 * 256 + test_b2
            pixel2 = cluster_centers[colors.index(min(colors,
                                                     key=lambda col:
                                                     abs(col - test_pixel2)))]
            new2[y + 1][x - 1][0] = pixel2[0]
            new2[y + 1][x - 1][1] = pixel2[1]
            new2[y + 1][x - 1][2] = pixel2[2]

            # bottom neighbour
            test_r3 = new[y][x + 1][0] + err_r * 5 / 16.0
            test_g3 = new[y][x + 1][1] + err_g * 5 / 16.0
            test_b3 = new[y][x + 1][2] + err_b * 5 / 16.0
            test_pixel3 = test_r3 * 65536 + test_g3 * 256 + test_b3
            pixel3 = cluster_centers[colors.index(min(colors,
                                                     key=lambda col:
                                                     abs(col - test_pixel3)))]
            new2[y + 1][x][0] = pixel3[0]
            new2[y + 1][x][1] = pixel3[1]
            new2[y + 1][x][2] = pixel3[2]

            # bottom right-hand corner neighbour
            test_r4 = new[y][x + 1][0] + err_r * 1 / 16.0
            test_g4 = new[y][x + 1][1] + err_g * 1 / 16.0
            test_b4 = new[y][x + 1][2] + err_b * 1 / 16.0
            test_pixel4 = test_r4 * 65536 + test_g4 * 256 + test_b4
            pixel4 = cluster_centers[colors.index(min(colors,
                                                     key=lambda col:
                                                     abs(col - test_pixel4)))]
            new2[y + 1][x + 1][0] = pixel4[0]
            new2[y + 1][x + 1][1] = pixel4[1]
            new2[y + 1][x + 1][2] = pixel4[2]

    # creating an image from the array
    new3 = Image.fromarray(new2.astype('uint8'))

    # saving the image as 'input_name' + '_new.png'
    new3.save('%s_new.png' % (str(pic)))


if __name__ == '__main__':
    if args.num_colors:
        c_q(args.file, args.num_colors)
    else:
        c_q(args.file)
