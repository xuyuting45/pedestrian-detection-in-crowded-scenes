#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from my_nms import cpu_re_nms

import _init_paths

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'person')

NETS = {'vgg16': ('VGG16','vgg16_faster_rcnn_iter_30000.caffemodel'),
    'zf': ('ZF','zf_faster_rcnn_split4bronze_iter3w.caffemodel')}

def draw_detections(im, dets, image_name, thresh=0.5,colour=(0,0,255)):
    """Draw detected bounding boxes."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    img_n=image_name.split('.')[0]
    inds = np.where(dets[:, -1] >= thresh)[0]
    
    if len(inds) == 0:
        print 'there is no person'
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),colour,2)        
        #draw the confidence
        str3 =('{:.3f}').format(score)
        cv2.putText(im,str3,(bbox[0],bbox[1]), font, 1,colour,1,1)

def demo(net, image_name):
     """Detect object classes in an image using pre-computed object proposals."""
     
     im_file = os.path.join(cfg.DATA_DIR, 'bronze-images', image_name)
     im = cv2.imread(im_file)
    
     out_dir = "/home/xyt/py-faster-rcnn/result_vis/result-bronze-similar0.8/" 
    
     # Detect all object classes and regress object bounds
     timer = Timer()
     timer.tic()
     scores, boxes = im_detect(net, im)
     timer.toc()
     print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
     CONF_THRESH = 0.5
     NMS_THRESH = 0.3
     cls_ind = 1 # because we skipped background
     cls_boxes = boxes[:, 4:8]
     cls_scores = scores[:, cls_ind]
     dets = np.hstack((cls_boxes,
                       cls_scores[:, np.newaxis])).astype(np.float32)
    
     draw_detections(im, dets, image_name, thresh=CONF_THRESH,colour=(0,255,0))
     keep=cpu_re_nms(dets,thresh=0.3,similarity=0.8)
     #keep = nms(dets, NMS_THRESH)
     dets = dets[keep, :]
     draw_detections(im, dets, image_name, thresh=CONF_THRESH,colour=(0,0,255))

     cv2.imwrite(out_dir+image_name, im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.RPN_POST_NMS_TOP_N=100	

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'trained_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    for i in range(1,501):
         print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' 
         im_name=str(i)+".jpg"
          # Load the demo image
         print 'Demo for data/demo/{}'.format(im_name)
         demo(net,im_name)
    
    print 'done!'