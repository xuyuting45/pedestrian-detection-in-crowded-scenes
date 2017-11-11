#!/usr/bin/env python

import numpy as np

def bbox_delta(ex_rois, gt_rois):
     ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
     ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
     ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
     ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
     gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
     gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
     gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
     gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
     targets_dx = np.abs((gt_ctr_x - ex_ctr_x) / ex_widths)
     targets_dy = np.abs((gt_ctr_y - ex_ctr_y) / ex_heights)
     targets_dxy= np.sqrt(targets_dx**2 + targets_dy**2)    
     targets_dw = np.abs((gt_widths / ex_widths)-1)
     targets_dh = np.abs((gt_heights / ex_heights)-1)    
     targets = np.vstack((targets_dxy, targets_dw, targets_dh)).transpose()
     #print targets    
     delta = np.sum(targets,axis=1)    
     return delta

def cpu_re_nms(dets,thresh=0.3,similarity=0.8):
     x1 = dets[:, 0]
     y1 = dets[:, 1]
     x2 = dets[:, 2]
     y2 = dets[:, 3]        
     scores = dets[:, 4]
     box=dets[:,:4]
     areas = (x2 - x1 + 1) * (y2 - y1 + 1)   
     order = scores.argsort()[::-1]
     print order
     keep = []
     while order.size > 0:
         i = order[0]
         fir = np.array([i])
         rest= order[1:]
         keep.append(i)
         xx1 = np.maximum(x1[i], x1[order[1:]])
         yy1 = np.maximum(y1[i], y1[order[1:]])
         xx2 = np.minimum(x2[i], x2[order[1:]])
         yy2 = np.minimum(y2[i], y2[order[1:]])
         w = np.maximum(0.0, xx2 - xx1 + 1)
         h = np.maximum(0.0, yy2 - yy1 + 1)
         inter = w * h
         ovr = inter / (areas[i] + areas[order[1:]] - inter)
         #print ovr,ovr.size
         inds = np.where(ovr <= thresh)[0]
         ##order = order[inds + 1]
         #print inds
         similar = bbox_delta(box[rest],box[fir])
         #print similar,similar.size
         label1= np.zeros((len(similar)),dtype=np.bool)
         add = np.where(similar > similarity)[0]
         label1[add]=True

         label2= np.zeros((len(ovr)),dtype=np.bool)
         label3= np.zeros((len(ovr)),dtype=np.bool)
         add = np.where(ovr > thresh)[0]
         label2[add]=True
         add = np.where(ovr < 0.5)[0]
         label3[add]=True
         view= label2 & label3
         add = np.where(view == True)[0]
	 print np.sort(similar[add])

         label =label1 & label2 & label3
         add = np.where(label == True)[0]
         #print label,add
         result=np.concatenate((inds,add))
         #print result
         order = order[result+1]
         #print order,'\n'
     return keep

if __name__ == '__main__':
      dets=np.array([[0,0,2,2,0.9],[0,0,1.8,1.8,0.8],[1,1,3,2,0.85],[1.1,1.1,2.1,3.1,0.82],[0,0,2,2,0.7]])
      print dets
      keep=cpu_re_nms(dets,0.3,0.1)
      print keep
