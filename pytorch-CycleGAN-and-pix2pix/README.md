## 接口总结

###### 7.23

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




