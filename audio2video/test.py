# import os, cv2, math, librosa
# import numpy as np
# import tensorflow as tf
# from src.approaches import *
# from utils import *

# landmarks_dir = '/home/wanghy/audio2video/data/face_npy'
# fl_origin_dir = '/home/summer_student/face-alignment-master/original_npy'
# audio_dir = '/home/wanghy/audio2video/data'
# energy_dir = '/home/wanghy/audio2video/data/energy'
# var_list_name = '/home/wanghy/audio2video/face_var.npy'
# colors = [(255, 144, 25), (50, 205, 50), (50, 205, 50), (208, 224, 63), (71, 99, 255), (71, 99, 255), (238, 130, 238), (238, 130, 238)]

# face_aver = get_average_face(landmarks_dir)
# var_list = np.load(var_list_name)

# # model_name = 'my_resnet_L2_audio_112'
# # model = Audio_Landmarks_block(model='resnet', model_save_dir='./model_save/{}'.format(model_name))
# model_name = 'multi_task_l2_4_3'
# model = Audio_Landmarks_Multi_Task(model='multi_task', model_save_dir='./model_save/{}'.format(model_name))
# model.load_model(epoch=200)

# def generate_landmarks_from_audio(f):
#     global landmarks_dir, fl_origin_dir, audio_dir, energy_dir, var_list, face_aver, model_name, model, colors
#     person = int(f[0]) - 1
#     file_name = f[:-4]
#     # get origin landmarks
#     fl_origin_path = os.path.join(fl_origin_dir, f)
#     fl_origin = np.load(fl_origin_path).reshape(-1, 68, 3)
#     len_fl_origin = fl_origin.shape[0]
#     # get landmarks after rotate
#     fl_rotated_path = os.path.join(landmarks_dir, f)
#     fl_rotated = np.load(fl_rotated_path).reshape(-1, 68, 3)
#     len_fl_rotated = fl_rotated.shape[0]
#     fl_norm = np.load(os.path.join('/home/wanghy/audio2video/data/face_npy_norm', f))
#     # get predict landmarks
#     audio_path = os.path.join(audio_dir, f)
#     prob = np.load(audio_path)
#     energy_path = os.path.join(energy_dir, f)
#     energy = np.load(energy_path)
#     audio_inputs, energy_inputs = audio_split(prob, energy)
#     audio_inputs = tf.convert_to_tensor(audio_inputs, dtype=tf.float32)
#     energy_inputs = tf.convert_to_tensor(energy_inputs, dtype=tf.float32)
#     fl_pred = model.single_eval(audio_inputs, energy_inputs).numpy().reshape(-1, 204)
#     fl_pred = np.concatenate([np.repeat(fl_pred[0:1], 4, axis=0), fl_pred], axis=0)
#     len_pred = fl_pred.shape[0]
#     # process landmarks
#     length = min(len_fl_origin, len_fl_rotated, len_pred)
#     fl_origin = fl_origin[:length]
#     fl_rotated = landmarks_bias(fl_rotated[:length], bias=fl_origin)
#     fl_pred = landmarks_bias(landmarks_restore(landmark=fl_pred[:length], average=face_aver[person]), bias=fl_origin)
#     # basic settings for video
#     video = cv2.VideoCapture('/home/summer_student/work/dataset/{}.avi'.format(file_name))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fps = video.get(cv2.CAP_PROP_FPS)
#     size = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
#     # print('video fps:', fps)
#     # print('video duration:', video.get(7)/video.get(5))
#     # print('video origin length:', len_fl_origin)
#     # print('video predict length:', len_pred)
#     # print('video final length:', length)
#     # generate landmarks video
#     output_dir = './test_result/{}'.format(model_name)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if not os.path.exists(os.path.join(output_dir, 'test_{}'.format(file_name))):
#         os.makedirs(os.path.join(output_dir, 'test_{}'.format(file_name)))
#     out = cv2.VideoWriter(os.path.join(output_dir, 'test_{}.avi'.format(file_name)), fourcc, fps, (size[1] * 3, size[0]))
#     for i in range(length):
#         img = landmarks_to_image([fl_origin[i], fl_rotated[i], fl_pred[i]], size, colors)
#         cv2.imwrite('./test_result/{}/test_{}/{}.jpg'.format(model_name, file_name, i), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
#         out.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
#     # merge audio and video
#     os.system('ffmpeg -i ./test_result/{}/test_{}.avi -i ./data/{}.wav -vcodec copy -acodec copy ./test_result/{}/{}.avi'.format(model_name, file_name, file_name, model_name, file_name))
#     # print('audio feature length:', prob.shape[0])
#     # print('audio duration:', librosa.get_duration(filename=f'./data/{file_name}.wav'))

# f = '1_0000.npy'
# generate_landmarks_from_audio(f)

import os, torch
from utils import *
from API import audio_to_landmarks

file_name = '1_0000'
audio_path = f'/home/wanghy/audio2video/data/{file_name}.wav'
size = (255, 255)

# from . import audio_to_features
# prob_path, energy_path = audio_to_features(audio_path)
# print(prob_path, energy_path)
tmp_dir = 'audio2landmarks_tmp'
landmarks_dir = '/home/wanghy/audio2video/data/face_npy'
face_aver = get_average_face(landmarks_dir)

path = audio_to_landmarks(audio_path, size, face_aver[0], model_name='multi_task_l2_1', model_dir='')
print(path)
