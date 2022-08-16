import numpy as np
from numpy import *
import cv2
import glob
import time
import os
import argparse
import math
import  Net.MGANet as MGANet
import torch
import copy
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from natsort import natsorted

def png_tensor(png):
    #trans = T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),T.Grayscale()])
    trans = T.Compose([T.ToTensor()])
    png = trans(png)
    return png 

def get_data(one_filename,video_index,num_frame):

    one_filename_length = len(one_filename)
    data_Y = []
    data_Y0 = []
    data_Y1 = []
    data_Y2 = []

    for i in range(one_filename_length+1):
        if i == 0:
            data_37_filename = natsorted(glob.glob(one_filename[i]+'/*.png'))
            data_len = len(data_37_filename)
            for i_0 in range(num_frame+1):
                file_name = data_37_filename[i_0]
                img_l = Image.open(file_name)
                img_l = png_tensor(img_l)
                data_Y0.append(img_l)
        if i == 1:
            mask_37_filename = natsorted(glob.glob(one_filename[i] + '/*.png'))
            mask_len = len(mask_37_filename)
            for i_1 in range(num_frame+1):
                file_name = mask_37_filename[i_1]
                img_m = Image.open(file_name)
                img_m = png_tensor(img_m)
                data_Y1.append(img_m)
        if i == 2:
            label_37_filename = natsorted(glob.glob('../label/' + '*.png'))
            label_len = len(label_37_filename)
            for i_2 in range(num_frame+1):
                file_name = label_37_filename[i_2]
                img_h = Image.open(file_name)
                img_h = png_tensor(img_h)
                data_Y2.append(img_h)
    data_Y.append(data_Y0)
    data_Y.append(data_Y1)
    data_Y.append(data_Y2)

    return  data_Y

def test_batch(data_Y, start, batch_size=1):
   
    data_pre = (data_Y[0][start-1])
    data_cur = (data_Y[0][start])
    data_aft = (data_Y[0][start+1])
    mask     = (data_Y[1][start])
    label    = (data_Y[2][start])
    start+=1
    return  data_pre,data_cur,data_aft,mask,label,start

def PSNR(img1, img2):
    mse = np.mean( (img1.astype(np.float32) - img2.astype(np.float32)) ** 2 ).astype(np.float32)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description="MGANet_train")
parser.add_argument("--video_nums", default=1, type=int, help="Videos number (default: 0)")
parser.add_argument("--frame_nums", default=290, type=int, help="frame number of the video to test (default: 90)")
parser.add_argument("--result_path", default='./result_png/', type=str)
opts = parser.parse_args()

one_filename = np.sort(glob.glob('../training_set/' + '*'))
print(one_filename)
patch_size =[480,832]
net_G = MGANet.Gen_Guided_UNet(batchNorm=False,input_size=patch_size,is_training=True)
net_G = net_G.cuda()

criterion_MSE = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net_G.parameters(), lr=0.001)
transform = T.Compose([T.ToTensor()])
Tensor = torch.cuda.FloatTensor
for epoch in range(100):
    frame_nums = opts.frame_nums
    video_num=opts.video_nums
    for video_index in range(video_num):
        data_Y = get_data(one_filename,video_index=video_index,num_frame=opts.frame_nums+5)
        start =1
        nums =opts.frame_nums
        for itr in range(0, nums):
            print(start)           
            data_pre, data_cur, data_aft, mask, label, start = test_batch(data_Y=data_Y, start=start, batch_size=1)
           
            optimizer.zero_grad()
            data_pre_value_patch = data_pre.float().cuda()
            data_cur_value_patch = data_cur.float().cuda()
            data_aft_value_patch = data_aft.float().cuda()
            data_mask_value_patch = mask.float().cuda()
            data_label_value_patch = label.float().cuda()
            data_pre_value_patch = torch.unsqueeze(data_pre_value_patch,0)
            data_cur_value_patch = torch.unsqueeze(data_cur_value_patch,0)
            data_aft_value_patch = torch.unsqueeze(data_aft_value_patch,0)
            data_mask_value_patch = torch.unsqueeze(data_mask_value_patch,0)
            data_label_value_patch = torch.unsqueeze(data_label_value_patch,0)

            print(data_pre_value_patch.size())
            print(data_cur_value_patch.size())
            print(data_aft_value_patch.size())
            print(data_mask_value_patch.size())
            print(data_label_value_patch.size())

            loss=0
            img_4,img_3,img_2,img_1,fake_image = net_G(data_pre_value_patch,data_cur_value_patch,data_aft_value_patch,data_mask_value_patch)
            loss = criterion_MSE(fake_image,data_label_value_patch)

            loss.backward()
            optimizer.step()

            fake_image_numpy = fake_image.detach().cpu().numpy()
            fake_image_numpy = np.squeeze(fake_image_numpy)*255.0
            finally_image=np.squeeze(fake_image_numpy)

            data_cur_value_patch = data_cur_value_patch.detach().cpu().numpy()
            data_cur_value_patch = np.squeeze(data_cur_value_patch)*255.0
            data_cur_value_patch = np.squeeze(data_cur_value_patch)
            mask = mask.detach().cpu().numpy()
            mask_image = np.squeeze(mask)*255.0
            label = label.detach().cpu().numpy()
            label = np.squeeze(label)*255.0
            os.makedirs(opts.result_path+'/result_enhanced_data/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_mask/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_label/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_compression_data/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_cur_data/%02d'%(video_index+1),exist_ok = True)

            cv2.imwrite(opts.result_path+'/result_enhanced_data/%02d/%02d.png'%(video_index+1,itr+2),finally_image.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_cur_data/%02d/%02d.png'%(video_index+1,itr+2),data_cur_value_patch.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_mask/%02d/%02d.png'%(video_index+1,itr+2),mask_image.astype(np.uint8))
            data_cur = data_cur.detach().cpu().numpy()
            data_cur_image = (np.squeeze(data_cur)*255.0).astype(np.float32)
            cv2.imwrite(opts.result_path+'/result_label/%02d/%02d.png'%(video_index+1,itr+2),label.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_compression_data/%02d/%02d.png'%(video_index+1,itr+2),data_cur_image.astype(np.uint8))
            print(loss.item())
