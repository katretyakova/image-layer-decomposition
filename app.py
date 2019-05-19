import numpy as np
import time
import scipy
import json
import decomposition
from scipy.spatial import ConvexHull, Delaunay
import scipy.sparse
import PIL.Image as Image
from numpy import *

def composite_layers(layers):
    layers = asfarray(layers)

    out = 255 * ones(layers[0].shape)[:, :, :3]

    for layer in layers:
        out += layer[:, :, 3:]/255. * (layer[:, :, :3] - out)

    return out

def save_results(img, rgb_palette, mixing_weights, output_prefix):
    from PIL import Image

    layers = []
    for li, color in enumerate(rgb_palette):
        layer = ones((img.shape[0], img.shape[1], 4), dtype = uint8)
        layer[:, :, :3] = asfarray(color * 255.0).round().clip(0, 255).astype(uint8)
        layer[:, :, 3] = (mixing_weights[:, :, li] * 255.).round().clip(0, 255).astype(uint8)
        layers.append(layer)
        outpath = output_prefix + '-layer%02d.png' % li
        Image.fromarray(layer).save(outpath)
        print ('Saved layer:', outpath)
    
    composited = composite_layers(layers)
    composited = composited.round().clip(0, 255).astype(uint8)
    outpath = output_prefix + '-composite.png'
    Image.fromarray(composited).save(outpath)
    print ('Saved composite:', outpath)
    return composited

def main(filename, image):
    import os

    img = np.asfarray(image.convert('RGB'))/255.0

    X, Y = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    XY = np.dstack((X * 1.0 / img.shape[0], Y * 1.0 / img.shape[1]))
    data = np.dstack((img, XY))
    
    rgb_palette = decomposition.simplify(img, filename[:-4])
    print("palette size: ", len(rgb_palette))
    
    rgbxy_hull = ConvexHull(data.reshape((-1, 5)))
    
    rgb_weights = decomposition.rgb_weights(img.reshape((-1, 3))[rgbxy_hull.vertices].reshape((-1, 1, 3)), rgb_palette)
    rgbxy_weights = decomposition.rgbxy_weights(rgbxy_hull.points[rgbxy_hull.vertices], rgbxy_hull.points)

    weights = rgbxy_weights.dot(rgb_weights.reshape((-1, len(rgb_palette)))).reshape((img.shape[0], img.shape[1], -1)).clip(0, 1)
    
    output_prefix = os.path.join('output', filename[:-4])
    save_results(img, rgb_palette, weights, output_prefix)

import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    img = Image.open(filename)
    main(filename, img)
