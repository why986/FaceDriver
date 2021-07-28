import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('handle with image before test')
parser.add_argument('--image', dest='image_path', type=str)
parser.add_argument('--result', dest='result_path', type=str)
args = parser.parse_args()

if os.path.isfile(args.image_path):
    image = cv2.imread(args.image_path, 1)
    temp = image.copy()
    re = np.concatenate([temp, image], 1)
    cv2.imwrite(args.result_path, re)
else:
    print("wrong path")
