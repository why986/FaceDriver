import os, cv2
import numpy as np

def landmarks_connect(img, fls, colors):
    def points(i1, i2):
        return (int(fls[i1, 0]), int(fls[i1, 1])), (int(fls[i2, 0]), int(fls[i2, 1]))
    def connect(index, color, loop=False, lineWidth=2):
        for i in index:
            p1, p2 = points(i, i + 1)
            cv2.line(img, p1, p2, color, lineWidth)
        if loop:
            p1, p2 = points(index[0], index[-1] + 1)
            cv2.line(img, p1, p2, color, lineWidth)

    connect(range(0, 16),  color=(255, 144, 25))  # jaw
    connect(range(17, 21), color=(50, 205, 50))  # eye brow
    connect(range(22, 26), color=(50, 205, 50))
    connect(range(27, 35), color=(208, 224, 63))  # nose
    connect(range(36, 41), loop=True, color=(71, 99, 255))  # eyes
    connect(range(42, 47), loop=True, color=(71, 99, 255))
    connect(range(48, 59), loop=True, color=(238, 130, 238))  # mouth
    connect(range(60, 67), loop=True, color=(238, 130, 238))
    
    return img

def calc_var():
    landmarks_dir = './data/face_npy'
    feature_list = []
    for i in range(9):
        feature_list.append([])
    for root, _, files in os.walk(landmarks_dir):
        for f in files:
            file_name = os.path.join(root, f)
            if file_name[-3:] != 'npy' or f[1] != '_':
                continue
            person = int(f[0]) - 1
            data = np.load(file_name)
            if len(feature_list[person]):
                feature_list[person] = np.concatenate([feature_list[person], data.reshape(data.shape[0], -1)], axis=0)
            else:
                feature_list[person] = data.reshape(data.shape[0], -1)
    var_list = []
    for i in range(9):
        if len(feature_list[i]):
            t = np.std(feature_list[i], axis=0)
            t[t==0] = 1.
            var_list.append(t)
        else:
            var_list.append(np.ones(204))

    np.save('face_var.npy', np.array(var_list))

def get_average_face(landmarks_dir):
    face_aver = []
    for i in range(9):
        file_name = os.path.join(landmarks_dir, '{}.npy'.format(i+1))
        if os.path.exists(file_name):
            face_aver.append(np.load(file_name))
        else:
            face_aver.append(np.zeros(204))
    return face_aver

def landmarks_bias(landmark, bias):
    return landmark.reshape(-1, 68, 3) + np.repeat(bias[:, 30:31, :], 68, axis=1).reshape(-1, 68, 3)

def landmarks_restore(landmark, average, std=None):
    if std is None:
        fl = (landmark + average).reshape(-1, 68, 3)
    else:
        fl = (landmark * std + average).reshape(-1, 68, 3)
    return fl

def audio_split(audio_input, energy_input, audio_dim=256, energy_dim=1):
    audio_len = int(audio_input.shape[0])
    block_num = (audio_len - 1) // 64 + 1
    total_len = int((block_num - 1) * 64 + 80)

    pad_len = total_len - audio_len
    audio_pad = np.concatenate([audio_input, np.zeros((pad_len, audio_dim))], axis=0)
    energy_pad = np.concatenate([energy_input, np.zeros((pad_len, energy_dim))], axis=0)

    audio = []
    energy = []
    for i in range(block_num):
        frame = 64 * i
        audio.append(audio_pad[frame:frame+80, :])
        energy.append(energy_pad[frame:frame+80, :])
    return np.array(audio), np.array(energy)

def landmarks_to_image(landmarks_seq, size, colors):
    imgs = []
    for landmark in landmarks_seq:
        img = landmarks_connect(np.ones((size[0], size[1], 3))*255, landmark, colors)
        imgs.append(img)
    return np.concatenate(imgs, axis=1)

