import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('avi to png before combine')
parser.add_argument('--fold_avi', dest='fold_avi', help='directory for avi', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_png', dest='fold_png', help='directory for png', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--use_5', dest='use_5', help='only use 5_xx', action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

avi_list = os.listdir(args.fold_avi)
now_avi = 0
for i, avi in enumerate(avi_list):
    now_num = 0
    name = avi.split('/')[-1]
    if args.use_5 and not '5_' in name:
        continue
    print(now_avi, avi)
    video = cv2.VideoCapture(os.path.join(args.fold_avi, avi))
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(args.fold_png+'/'+name.split('.')[0]+'_'+str(now_num)+'.png', frame)
        now_num += 1
    now_avi += 1
