import os, argparse, cv2
import numpy as np
from super_resolution import SRModel

parser = argparse.ArgumentParser('aduio2video')
parser.add_argument('--file_name', dest='file_name', type=str, default='5_0001')
parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default='data/')
parser.add_argument('--landmarks_dir', dest='landmarks_dir', type=str, default='average_face/')
parser.add_argument('--model_name', dest='model_name', type=str, default='multi_task_l2_1')
parser.add_argument('--result_dir', dest='result_dir', type=str, default='result')
args = parser.parse_args()

# audio2video
os.chdir('audio2video')
tmp_dir = 'audio2landmarks_tmp'
os.system(f'python test.py --tmp_dir {tmp_dir} --file_name {args.file_name} --dataset_dir {args.dataset_dir} --landmarks_dir {args.landmarks_dir} --model_name {args.model_name}')

# preprocess
tmp2_dir = '../pytorch-CycleGAN-and-pix2pix/datasets/face/'+args.file_name
if not os.path.exists(tmp2_dir):
    os.mkdir(tmp2_dir)
if not os.path.exists(os.path.join(tmp2_dir, 'test')):
    os.mkdir(os.path.join(tmp2_dir, 'test'))
for root, _, files in os.walk(os.path.join(tmp_dir, args.file_name)):
    for f in files:
        file_name = os.path.join(root, f)
        image = cv2.imread(file_name, 1)
        temp = image.copy()
        re = np.concatenate([temp, image], 1)
        cv2.imwrite(os.path.join(tmp2_dir, 'test', f), re)

# pix2pix
os.chdir('../pytorch-CycleGAN-and-pix2pix')
tmp3_dir = f'results/face_pix2pix/{args.file_name}'
epochs = 10
os.system(f'python test.py --dataroot {tmp2_dir} --direction BtoA --model pix2pix --name face_pix2pix --epoch {epochs} --results_dir {tmp3_dir}')

# change face

# super resolution
os.chdir('..')
result_dir = args.result_dir
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
if not os.path.exists(os.path.join(result_dir, 'image')):
    os.mkdir(os.path.join(result_dir, 'image'))
output_path = os.path.join(result_dir, f'{args.file_name}.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 25
out = cv2.VideoWriter(output_path, fourcc, fps, (255, 255))

sr_model = SRModel(gpu_ids='0,1') # assume using gpu 0,1
for root, _, files in os.walk(os.path.join('pytorch-CycleGAN-and-pix2pix', tmp3_dir, f'face_pix2pix/test_{epochs}/images')):
    for f in files:
        if not 'fake' in f:
            continue
        img = cv2.imread(os.path.join(root, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sr_model.forward(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, 'image', f), img)
        out.write(img)
os.system(f'ffmpeg -y -i {output_path} -i audio2video/{os.path.join(args.dataset_dir,args.file_name)}.wav -c {output_path}')