# FaceDriver

### Dependencies
* Python 3.7
* ffmpeg 3.4.8
* pytorch 0.9.0

### 1.audio2video

- [deepspeech模型参数](https://cloud.tsinghua.edu.cn/f/fb061ff811fe44089462/?dl=1)
- [关键点生成模型参数](https://cloud.tsinghua.edu.cn/f/df7e6b31bd494efc9062/?dl=1)

调用API.py中的audio_to_landmarks函数，可完成从语音音频到人脸关键点的预测，具体接口详见audio2video/API.py。

### 2.pix2pix

#### 预训练模型

 [新闻联播人脸预训练模型](https://cloud.tsinghua.edu.cn/d/f59f3ea1fe3e4dd892ce/)



#### 关键点图片一个预处理

在pytorch-CycleGan-and-pix2pix/datasets下，依据模型需要改一点图片要求

```
python beforetest.py  --image ./face/test/5_0000_39.png  --result ./face/test/a.png
```



#### 关键点→人脸 pix2pix

在pytorch-CycleGan-and-pix2pix内，checkpoints/face_pix2pix内是预训练模型，datasets/face内是所有数据

```
python test.py --dataroot ./datasets/face --direction BtoA --model pix2pix --name face_pix2pix --epoch 10
```

### 3. 换脸 first order model

在first-order-model内

```
python demo.py  --config config/vox-adv-256.yaml --driving_video 驱动视频的地址 --source_image 输入图片的地址 --checkpoint ../vox-adv-cpk.pth.tar --relative --adapt_scale
```

[预训练模型](https://cloud.tsinghua.edu.cn/f/bba16a6f308d490ca0c0/?dl=1)


### 4. Face Super Resolution

This module is based on https://github.com/ewrfcas/Face-Super-Resolution

#### Setup

* Download [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1u3h3nX5f_w-HJV8Nd1zwqc3uTnVja5Ol).
* Download pretrained generator weights [90000_G.pth](https://drive.google.com/open?id=1CZkLZPtbJepgksCM93MvsY7NgqnEZSvk).
* Put the above files into `data` folder.

#### Example

```python3
import cv2
from super_resolution import SRModel

sr_model = SRModel(gpu_ids='0,1') # assume using gpu 0,1

# Read image from file
# - Remember to convert image format from BGR to RGB!
img = cv2.imread('input.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run the model
# - The model accepts numpy.ndarray (RGB format) as input, as well as output.
img = sr_model.forward(img)

# Write image to file for preview
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.png', img)
```
