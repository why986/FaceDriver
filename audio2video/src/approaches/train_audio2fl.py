import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from ..models.ResNet import ResNet_Audio2FL
import os

FACE_POINTS_NUM  = 48*3
MOUTH_POINTS_NUM = 20*3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyMetric(object):
    def __init__(self, method='l1'):
        super(MyMetric, self).__init__()
        self.method = method
        self.metric = 0
    
    def eval(self, pred, true):
        if self.method == 'l1':
            self.metric = F.l1_loss(pred, true)
        elif self.method == 'l2':
            self.metric = F.mse_loss(pred, true)
    
    def result(self):
        return self.metric


class MyLoss(nn.Module):
    def __init__(self, method='l1', weights=[1,1,1]):
        '''
            method -- l1 or l2
            weights -- out : h : l
        '''
        super(MyLoss, self).__init__()
        self.w = weights
        self.method = method
    
    def forward(self, y_pred, h_pred, l_pred, y_true, h_true, l_true):
        if self.method == 'l1':
            loss1 = F.l1_loss(y_pred, y_true)
            loss2 = F.l1_loss(h_pred, h_true)
            loss3 = F.l1_loss(l_pred, l_true)
        elif self.method == 'l2':
            loss1 = F.mse_loss(y_pred, y_true)
            loss2 = F.mse_loss(h_pred, h_true)
            loss3 = F.mse_loss(l_pred, l_true)
        loss = self.w[0] * loss1 + self.w[1] * loss2 + self.w[2] * loss3
        return loss


class Audio_Landmarks_block(object):
    def __init__(self, dataset=[], loss='l1', loss_weights=[1,1,1], lr=1e-3, epochs=100, model_save_dir='./model_save', jump_flag=False):
        self.model_save_dir = model_save_dir
        self.dataset = dataset
        self.jump_flag = jump_flag

        self.model_face = ResNet_Audio2FL(num_output=FACE_POINTS_NUM).to(device)
        self.loss_face = MyLoss(method=loss, weights=loss_weights).to(device)
        self.optim_face =  [optim.Adam(self.model_face.parameters(), lr=lr), optim.Adam(self.model_face.parameters(), lr=lr/10),
                            optim.Adam(self.model_face.parameters(), lr=lr/100), optim.Adam(self.model_face.parameters(), lr=lr/1000)]
        self.train_metric_face = MyMetric(method=loss)

        self.model_mouth = ResNet_Audio2FL(num_output=MOUTH_POINTS_NUM).to(device)
        self.loss_mouth = MyLoss(method=loss, weights=loss_weights).to(device)
        self.optim_mouth = [optim.Adam(self.model_mouth.parameters(), lr=lr), optim.Adam(self.model_mouth.parameters(), lr=lr/10),
                            optim.Adam(self.model_mouth.parameters(), lr=lr/100), optim.Adam(self.model_mouth.parameters(), lr=lr/1000)]
        self.train_metric_mouth = MyMetric(method=loss)

        self.epochs = epochs

        # if torch.cuda.is_available():
        #     self.model_face = self.model_face.cuda()
        #     self.loss_face = self.loss_face.cuda()
        #     self.model_mouth = self.model_mouth.cuda()
        #     self.loss_mouth = self.loss_mouth.cuda()
    
    def train(self, model_save=True, model_name='audio2fl_model_weights'):
        if model_save and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        def face_train_step(opt_idx, audio_input, energy_input, face_true, h_true, l_true, FLAG):
            self.model_face.train()
            face_pred, h_pred, l_pred = self.model_face(audio_input, energy_input)
            loss = self.loss_face(face_pred, h_pred, l_pred, face_true, h_true, l_true)
            if FLAG: loss *= 50
            self.train_metric_face.eval(face_pred, face_true)

            self.optim_face[opt_idx].zero_grad()
            loss.backward()
            self.optim_face[opt_idx].step()

        def mouth_train_step(opt_idx, audio_input, energy_input, mouth_true, h_true, l_true, FLAG):
            self.model_mouth.train()
            mouth_pred, h_pred, l_pred = self.model_mouth(audio_input, energy_input)
            loss = self.loss_mouth(mouth_pred, h_pred, l_pred, mouth_true, h_true, l_true)
            if FLAG: loss *= 50
            self.train_metric_mouth.eval(mouth_pred, mouth_true)

            self.optim_mouth[opt_idx].zero_grad()
            loss.backward()
            self.optim_mouth[opt_idx].step()
        
        iter_cnt = 0
        for epoch in range(self.epochs):
            opt_idx = epoch // 50

            for energy_input, audio_input, output_true, h_true, l_true in self.dataset:
                iter_cnt += 128
                face_true, mouth_true = torch.split(output_true, [FACE_POINTS_NUM, MOUTH_POINTS_NUM], dim=2)
                face_h, mouth_h = torch.split(h_true, [FACE_POINTS_NUM, MOUTH_POINTS_NUM], dim=2)
                face_l, mouth_l = torch.split(l_true, [FACE_POINTS_NUM, MOUTH_POINTS_NUM], dim=2)
                face_train_step(opt_idx, audio_input, energy_input, face_true, face_h, face_l, (iter_cnt>3000) and self.jump_flag)
                mouth_train_step(opt_idx, audio_input, energy_input, mouth_true, mouth_h, mouth_l, (iter_cnt>3000) and self.jump_flag)

            print(f'Epoch {epoch+1}, train face_loss: {self.train_metric_face.result()}, mouth_loss: {self.train_metric_mouth.result()}')

            if model_save and (epoch + 1)%50 == 0:
                torch.save(self.model_face.state_dict(), os.path.join(self.model_save_dir, f'{model_name}_face_{epoch + 1}.pth'))
                torch.save(self.model_mouth.state_dict(), os.path.join(self.model_save_dir, f'{model_name}_mouth_{epoch + 1}.pth'))

    def load_model(self, epoch=100, model_name='audio2fl_model_weights'):
        self.model_face.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{model_name}_face_{epoch}.pth')))
        self.model_mouth.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{model_name}_mouth_{epoch}.pth')))
    
    def single_eval(self, audio_input, energy_input):
        self.model_face.eval()
        self.model_mouth.eval()
        face_pred, _, _ = self.model_face(audio_input, energy_input)
        mouth_pred, _, _ = self.model_mouth(audio_input, energy_input)
        output = torch.cat((face_pred, mouth_pred), dim=2)
        return output