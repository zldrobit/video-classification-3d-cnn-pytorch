import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import time
from PIL import Image

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)

import cv2

if __name__=="__main__":
    opt = parse_opts()
    assert opt.mode == 'score'
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    print(opt.arch, model_data['arch'])
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 800, 600)
    x1, y1 = 0, 0
    x2, y2 = 800, 600

    clip = torch.randn([1, 3, opt.sample_duration, opt.sample_size, opt.sample_size], 
                        dtype=torch.float, requires_grad=False)
    color = (144,238,144)

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    clip_results = dict()
    vdo = cv2.VideoCapture() 
    # vdo.open(opt.webcam_ind)
    vdo.open(0)
    iframe = 0
    accum_fps = 0
    # warm-up
    print("Warming up.")
    for i in range(10):
        with torch.no_grad():
            outputs = model(clip)

    while vdo.grab():
        start = time.time()
        _, cv2_im = vdo.retrieve()
        rgb_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb_im)
        im = spatial_transform(pil_im)
        if iframe < opt.sample_duration:
            clip[0, :, iframe, :, :] = im
        else:
            clip[0, :, :-1, :, :] = clip[0, :, 1:, :, :]
            clip[0, :, -1, :, :] = im
        with torch.no_grad():
            outputs = model(clip)
        video_outputs = outputs.cpu().data
        _, max_indices = video_outputs.max(dim=1)
        label = clip_results['label'] = class_names[max_indices[0]]
        clip_results['scores'] = video_outputs[0].tolist()
        end = time.time()
        fps = 1 / (end - start)
        accum_fps += fps
        avg_fps = accum_fps / (iframe + 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(cv2_im,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(cv2_im,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        cv2.imshow("test",cv2_im)
        cv2.waitKey(1)
        print("frame time: {}s, fps: {}, avg fps : {}".format(end - start, fps,  avg_fps))
        iframe += 1

