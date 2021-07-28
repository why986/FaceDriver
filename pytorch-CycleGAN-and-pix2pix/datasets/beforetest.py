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
    if os.path.isdir(args.image_path):
        for root, _, files in os.walk(args.image_path):
            for f in files:
                file_name = os.path.join(root, f)
                image = cv2.imread(file_name, 1)
                temp = image.copy()
                re = np.concatenate([temp, image], 1)
                cv2.imwrite(file_name, re)
    else:
        print("wrong path")
