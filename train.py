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

def yuv_import(filename, dims ,startfrm,numframe):
    fp = open(filename, 'rb')
    frame_size = np.prod(dims) * 3 / 2
    fp.seek(0, 2)

    ps = fp.tell()
    totalfrm = int(ps // frame_size)
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    assert startfrm+numframe<=totalfrm

    Y = np.zeros(shape=(numframe, 1,dims[0], dims[1]), dtype=np.uint8, order='C')
    U = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint8, order='C')
    V = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint8, order='C')

    fp.seek(int(frame_size * startfrm), 0)
    for i in range(startfrm,startfrm+numframe):
        for m in range(dims[0]):
            for n in range(dims[1]):
                Y[i-startfrm,0, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                U[i-startfrm,0, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                V[i-startfrm,0, m, n] = ord(fp.read(1))

    fp.close()
    Y = Y.astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    return Y, U, V

def get_w_h(filename):
    width = int((filename.split('x')[0]).split('_')[-1])
    height = int((filename.split('x')[1]).split('_')[0])
    print(width)
    print(height)
    return (height,width)

def get_data(one_filename,video_index,num_frame,startfrm_position):

    one_filename_length = len(one_filename)
    data_Y = []
    for i in range(one_filename_length+1):
        if i == 0:
            data_37_filename = np.sort(glob.glob(one_filename[i]+'/*.yuv'))
            data_37_filename_length = len(data_37_filename )
            for i_0 in range(video_index,video_index+1):
                file_name = data_37_filename[i_0]
                dims = get_w_h(filename=file_name)
                data_37_filename_Y,data_37_filename_U,data_37_filename_V = yuv_import(filename=file_name, dims=dims ,startfrm=startfrm_position,numframe=num_frame)
                data_Y.append(data_37_filename_Y)
               
        if i == 1:
            mask_37_filename = np.sort(glob.glob(one_filename[i] + '/*.yuv'))
            mask_37_filename_length = len(mask_37_filename)
            for i_1 in range(video_index,video_index+1):
                file_name = mask_37_filename[i_1]
                dims = get_w_h(filename=file_name)
                mask_37_filename_Y, mask_37_filename_U, mask_37_filename_V = yuv_import(filename=file_name, dims=dims,startfrm=startfrm_position, numframe=num_frame)
                data_Y.append(mask_37_filename_Y)
        if i == 2:
            label_37_filename = np.sort(glob.glob('../label1/' + '*.yuv'))
            label_37_filename_length = len(label_37_filename)
            for i_2 in range(video_index,video_index+1):
                file_name = label_37_filename[i_2]
                dims = get_w_h(filename=file_name)
                label_37_filename_Y, label_37_filename_U, label_37_filename_V = yuv_import(filename=file_name, dims=dims,startfrm=startfrm_position, numframe=num_frame)
                data_Y.append(label_37_filename_Y)
               
    return  data_Y

def test_batch(data_Y, start, batch_size=1):

    data_pre = (data_Y[0][start-1:start,...])/255.0
    data_cur = (data_Y[0][start:start+1,...])/255.0
    data_aft = (data_Y[0][start+1:start+2,...])/255.0

    mask     = (data_Y[1][start:start+1,...])/255.0
    label    = (data_Y[2][start:start+1,...])

    start+=1
    print(len(data_Y[0]))
    #print(data_Y[1])
    #print(data_Y[2])
    return  data_pre,data_cur,data_aft,mask,label,start

def PSNR(img1, img2):
    mse = np.mean( (img1.astype(np.float32) - img2.astype(np.float32)) ** 2 ).astype(np.float32)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description="MGANet_train")
parser.add_argument("--video_nums", default=2, type=int, help="Videos number (default: 0)")
parser.add_argument("--frame_nums", default=29, type=int, help="frame number of the video to test (default: 90)")
parser.add_argument("--startfrm_position", default=9, type=int, help="start frame position in one video (default: 0)")
parser.add_argument("--result_path", default='./result_train/', type=str)
opts = parser.parse_args()
print(torch.cuda.is_available())

one_filename = np.sort(glob.glob('../training/' + '*'))
print(one_filename)
patch_size =[240,416]
net_G = MGANet.Gen_Guided_UNet(batchNorm=False,input_size=patch_size,is_training=True)
net_G = net_G.cuda()

criterion_MSE = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net_G.parameters(), lr=0.001)
transform = T.Compose([T.ToTensor()])
Tensor = torch.cuda.FloatTensor
for epoch in range(100):

    video_num=opts.video_nums
    for video_index in range(video_num):
        data_Y = get_data(one_filename,video_index=video_index,num_frame=opts.frame_nums+5,startfrm_position=opts.startfrm_position)
        start =1
        nums =opts.frame_nums
        for itr in range(0, nums):           
            data_pre, data_cur, data_aft, mask, label, start = test_batch(data_Y=data_Y, start=start, batch_size=1)
           
            optimizer.zero_grad()
            data_pre_value_patch = torch.from_numpy(data_pre).float().cuda()
            data_cur_value_patch = torch.from_numpy(data_cur).float().cuda()
            data_aft_value_patch = torch.from_numpy(data_aft).float().cuda()
            data_mask_value_patch = torch.from_numpy(mask).float().cuda()
            data_label_value_patch = torch.from_numpy(label).float().cuda()
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
           
            fake_image_numpy = np.squeeze(fake_image_numpy)

            finally_image=np.squeeze(fake_image_numpy)
            data_cur_value_patch = data_cur_value_patch.detach().cpu().numpy()
            data_cur_value_patch = np.squeeze(data_cur_value_patch)*255.0
            data_cur_value_patch = np.squeeze(data_cur_value_patch)
            mask_image = np.squeeze(mask)*255.0
            os.makedirs(opts.result_path+'/result_enhanced_data/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_mask/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_label/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_compression_data/%02d'%(video_index+1),exist_ok = True)
            os.makedirs(opts.result_path+'/result_cur_data/%02d'%(video_index+1),exist_ok = True)

            cv2.imwrite(opts.result_path+'/result_enhanced_data/%02d/%02d.png'%(video_index+1,itr+2),finally_image.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_cur_data/%02d/%02d.png'%(video_index+1,itr+2),data_cur_value_patch.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_mask/%02d/%02d.png'%(video_index+1,itr+2),mask_image.astype(np.uint8))
            data_cur_image = (np.squeeze(data_cur)*255.0).astype(np.float32)
            label = np.squeeze(label).astype(np.float32)
            cv2.imwrite(opts.result_path+'/result_label/%02d/%02d.png'%(video_index+1,itr+2),label.astype(np.uint8))
            cv2.imwrite(opts.result_path+'/result_compression_data/%02d/%02d.png'%(video_index+1,itr+2),data_cur_image.astype(np.uint8))
            print(loss.item())
