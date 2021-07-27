import os
import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import transform as trans
from PIL import Image
from .models.SRGAN_model import SRGANModel

class SRModel:
    _transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])])

    def __init__(self, gpu_ids=None, data_path=None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data')
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(os.path.join(data_path, 'shape_predictor_68_face_landmarks.dat'))
        self.args = SRModelArgs()
        self.args.gpu_ids = gpu_ids
        self.args.pretrain_model_G = os.path.join(data_path, '90000_G.pth')
        self.sr_model = SRGANModel(self.args, is_train=False)
        self.sr_model.load()

    """
    img     - ndarray, rgb format
    @return - ndarray, rgb format
    """
    def forward(self, img, resize_to=(1024, 1024), padding=0.5, moving=0.1):
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)
        img_aligned, M = self.dlib_detect_face(img, padding=padding, image_size=(128, 128), moving=moving)
        input_img = torch.unsqueeze(self._transform(Image.fromarray(img_aligned)), 0)
        self.sr_model.var_L = input_img.to(self.sr_model.device)
        self.sr_model.test()
        output_img = self.sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        rec_img = self.face_recover(output_img, M * 4, img)
        return rec_img

    def face_recover(self, img, M, ori_img):
        # img:rgb, ori_img:bgr
        # dst:rgb
        dst = ori_img.copy()
        cv2.warpAffine(img, M, (dst.shape[1], dst.shape[0]), dst,
                    flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
        return dst


    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords


    def dlib_alignment(self, img, landmarks, padding=0.25, size=128, moving=0.0):
        x_src = np.array([0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                        0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                        0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                        0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                        0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                        0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                        0.553364, 0.490127, 0.42689])
        y_src = np.array([0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                        0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                        0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                        0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                        0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                        0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                        0.784792, 0.824182, 0.831803, 0.824182])
        x_src = (padding + x_src) / (2 * padding + 1)
        y_src = (padding + y_src) / (2 * padding + 1)
        y_src += moving
        x_src *= size
        y_src *= size

        src = np.concatenate([np.expand_dims(x_src, -1), np.expand_dims(y_src, -1)], -1)
        dst = landmarks.astype(np.float32)
        src = np.concatenate([src[10:38, :], src[43:48, :]], axis=0)
        dst = np.concatenate([dst[27:55, :], dst[60:65, :]], axis=0)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)

        return warped, M


    def dlib_detect_face(self, img, image_size=(128, 128), padding=0.25, moving=0.0):
        dets = self.dlib_detector(img, 0)
        if dets:
            if isinstance(dets, dlib.rectangles):
                det = max(dets, key=lambda d: d.area())
            else:
                det = max(dets, key=lambda d: d.rect.area())
                det = det.rect
            face = self.sp(img, det)
            landmarks = self.shape_to_np(face)
            img_aligned, M = self.dlib_alignment(img, landmarks, size=image_size[0], padding=padding, moving=moving)

            return img_aligned, M
        else:
            return None

class SRModelArgs:
    gpu_ids = None
    batch_size = 32
    lr_G = 1e-4
    weight_decay_G = 0
    beta1_G = 0.9
    beta2_G = 0.99
    lr_D = 1e-4
    weight_decay_D = 0
    beta1_D = 0.9
    beta2_D = 0.99
    lr_scheme = 'MultiStepLR'
    niter = 100000
    warmup_iter = -1
    lr_steps = [50000]
    lr_gamma = 0.5
    pixel_criterion = 'l1'
    pixel_weight = 1e-2
    feature_criterion = 'l1'
    feature_weight = 1
    gan_type = 'ragan'
    gan_weight = 5e-3
    D_update_ratio = 1
    D_init_iters = 0

    print_freq = 100
    val_freq = 1000
    save_freq = 10000
    crop_size = 0.85
    lr_size = 128
    hr_size = 512

    which_model_G = 'RRDBNet'
    G_in_nc = 3
    out_nc = 3
    G_nf = 64
    nb = 16

    which_model_D = 'discriminator_vgg_128'
    D_in_nc = 3
    D_nf = 64

    pretrain_model_G = '90000_G.pth'
    pretrain_model_D = None