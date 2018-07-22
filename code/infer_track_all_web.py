#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

import pycocotools.mask as mask_util
import numpy as np
import json

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--gap-threshold',
        dest='gap_threshold',
        help='Merge hand event by gap threshold',
        default='3',
        type=int
    )
    parser.add_argument(
        '--interested-objects',
        dest='interested_objects',
        help='interested objects',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

a = 0
thresh = 0.7
object_list = []

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 2
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    im_list = []
    if os.path.isdir(args.im_or_folder):
        path, dirs, files = os.walk(args.im_or_folder).next()
        file_count = len(files)
        for count in range(file_count):
            im_list.append(args.im_or_folder + ('/%d.jpg') % count)
#         im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]
        
        
    whichObject_strEndFrame_all = {}
    whichObject_strEndFrame = {}
    in_hand_event = False
    gap_count = 0
    started_frame = 0
    ended_frame = 0
    S = {}
    S_prime = []
    for img_i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        #logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        #for k, v in timers.items():
        #    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        #if i == 0:
        #    logger.info(
        #        ' \ Note: inference on the first image will be slower than the '
        #        'rest (caches and auto-tuning need to warm up)'
        #    )
        
        
        #Visualizaion result and save into images
#         vis_utils.vis_one_image(
#             im = im[:, :, ::-1],  # BGR -> RGB for visualization
#             im_name = im_name,
#             output_dir = args.output_dir,
#             boxes = cls_boxes,
#             segms = cls_segms,
#             keypoints = cls_keyps,
#             dataset=dummy_coco_dataset,
#             box_alpha=0.3,
#             show_class=True,
#             thresh=0.7,
#             kp_thresh=2,
#             ext='jpg'
#         )
        
        ######################################Show Mask###################
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
                cls_boxes, cls_segms, cls_keyps)
        if segms is not None:
            masks = mask_util.decode(segms)
        
#         # Display in largest to smallest order to reduce occlusion
#         areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#         sorted_inds = np.argsort(-areas)
# print (boxes[:, 4])
#             print (boxes.shape)
#     #         print(sorted_inds)
#             print (classes)
#             print ([i for i, j in enumerate(classes) if j == 40])
#             print (dummy_coco_dataset.classes[1])

        #sign 1 = person, 40 = bottle, 74 = book
        #Get how many bottles in first frame
        interestObjects = str(args.interested_objects).split(',')
        
        if (img_i == 20):
            global object_list
            for interestObject in interestObjects:
                object_list.append(dummy_coco_dataset.classes[int(interestObject)])
            whichObject_strEndFrame_all['object_list'] = [ob for ob in object_list]
        
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        scores = []
        classes_new = []
        for i in sorted_inds:
            classes_new.append(classes[i])
            scores.append(boxes[i, -1])
        
        classes = classes_new
        print(classes)
        print(scores)
           
        if 1 not in classes and not in_hand_event:
            continue
            
        elif 1 not in classes and in_hand_event:
            if gap_count >= args.gap_threshold:
                in_hand_event = False
                ended_frame = img_i
                
                object_indexs = []
                for inobject in interestObjects:
                    inobject_i = int(inobject)
                    indextem=[]
                    for index, ob in enumerate(classes):
                        if ob==inobject_i:
                            indextem.append(index)
                    for index in indextem:
                        if scores[index] > thresh:
                            object_indexs.append(index)
                
                for i in object_indexs:
                    mask = masks[:, :, i]
                    S_prime.append(mask)
                    
                
                WhichIsTaken, object_list = WhichIsTaken_fun(S, S_prime, object_list)
                S = {}
                S_prime = []
                class_name = dummy_coco_dataset.classes[WhichIsTaken]
                whichObject_strEndFrame[class_name] = [started_frame, ended_frame]
                # whichObject_strEndFrame[len(whichObject_strEndFrame)] = {WhichIsTaken : [started_frame, ended_frame]}
                
            else:    
                gap_count += 1
                continue
            
        elif 1 in classes and not in_hand_event:
            gap_count = 0    
            in_hand_event = True
            started_frame = img_i
            object_indexs = []
            for inobject in interestObjects:
                inobject_i = int(inobject)
                indextem=[]
                for index, ob in enumerate(classes):
                    if ob==inobject_i:
                        indextem.append(index)
                for index in indextem:
                    if scores[index] > thresh:
                        object_indexs.append(index)
            for i in object_indexs:
                mask = masks[:, :, i]
                S[classes[i]] = mask
#             bottle_median_dict = dict()
#             for i in bottle_indexs:
#                 mask = masks[:, :, i]
#                 #remenber i is along x-axis, j is along y-axis
#                 mask_index_i, mask_index_j = np.where(mask==1)
#                 mask_index_i_median = np.median(mask_index_i).astype(np.int64)
#                 mask_index_j_median = np.median(mask_index_j).astype(np.int64)
#                 bottle_median_dict[i] = (mask_index_i_median, mask_index_j_median)
            
#             bottle_median_sort = sorted(bottle_median_dict.items(), key=lambda x: x[1][1])

#             for index, bottle_median in enumerate(bottle_median_sort):
#                 label = ('bottle_%d' % index)
#                 mask = masks[:, :, bottle_median[0]]
                
                #save image
                #  cv2.imwrite(args.output_dir + ('/%s.jpg' % label), mask)
                
            continue
                
        elif 1 in classes and in_hand_event:
            continue
            
        
#         Objects_minXor = {}
#         for object_name, object_mask in S.iteritems():
#             result_xor = []
#             for object_prime_mask in S_prime:
#                 mask_xor = np.bitwise_xor(object_mask, object_prime_mask)
#                 result_xor.append(mask_xor.sum())
#             Objects_minXor[object_name] = min(result_xor)
        
#         Objects_minXor_sort = sorted(Objects_minXor.items(), key=lambda x: x[1], reverse=True)
        
#         # print(Objects_minXor_sort)
        
#         # WhichIsTaken_relative = Objects_minXor_sort[0][0]
#         WhichIsTaken_list = []
#         for which in [which for which, xor in Objects_minXor_sort if xor>25000]:
#             try:
#                 WhichIsTaken_list.append(bottle_list.pop(int(which.split('_')[1])))
#             except:
#                 WhichIsTaken_list.append(bottle_list.pop(int(which.split('_')[1])-1))
#         WhichIsTaken = ','.join(str(x) for x in WhichIsTaken_list)
# #         try:
# #             WhichIsTaken = bottle_list.pop(int(WhichIsTaken_relative.split('_')[1]))
# #         except:
# #             WhichIsTaken = bottle_list.pop(int(WhichIsTaken_relative.split('_')[1])-1)
#         #Clear S and S_prime for next hand event
        
    
    
    whichObject_strEndFrame_all['taken'] = whichObject_strEndFrame
#     bottle_all = whichObject_strEndFrame_all[0]
#     for key in whichObject_strEndFrame_all.keys()
    
    #Save final result from which frame to which frame and which object is taken
    with open(args.output_dir + ('/output.json'), "w") as f:
        json.dump(whichObject_strEndFrame_all, f)
#         try:
#             label = dummy_coco_dataset.classes[classes[i]]
#         except:
#             label = classes[i]
        
        
        
#         forjson_object = {}
#         for i in sorted_inds:
#             mask = masks[:,:,i]
            
#             if (len(np.unique(mask)) == 1):
#                 continue
#             #Get center of mask
#             mask_index_i, mask_index_j = np.where(mask==1)
#             mask_index_i_median = np.median(mask_index_i).astype(np.int64)
#             mask_index_j_median = np.median(mask_index_j).astype(np.int64)
            
#             # Especially Mask center
#             # mask[mask_index_i_median, mask_index_j_median] = 0
#             # np.place(mask, mask==1, [255])
            
#             try:
#                 label = dummy_coco_dataset.classes[classes[i]]
#             except:
#                 label = classes[i]
                
#             if label != 'bottle':
#                 continue
#             label = label+ ('_%d' % i)
#             cv2.imwrite(args.output_dir + ('/%s.jpg' % label), mask)
            
            
#             img_name = im_name.rsplit('/')[-1]
#             forjson_object[label] = [mask_index_j_median, mask_index_i_median]
#             forjson_all[img_name] = forjson_object
            
#     with open(args.output_dir + ('/output.json'), "w") as f:
#         json.dump(forjson_all, f)

def WhichIsTaken_fun(S, S_prime, object_list):
    Objects_minXor = {}
    for object_name, object_mask in S.iteritems():
        result_xor = []
        for object_prime_mask in S_prime:
            mask_xor = np.bitwise_xor(object_mask, object_prime_mask)
            result_xor.append(mask_xor.sum())
        if not result_xor:
            Objects_minXor[object_name] = 30000
        else:
            Objects_minXor[object_name] = min(result_xor)
            
    Objects_minXor_sort = sorted(Objects_minXor.items(), key=lambda x: x[1], reverse=True)
    WhichIsTaken_list = []
    
#     #XOR > pixel
#     for which in [which for which, xor in Objects_minXor_sort if xor>25000]:
#         try:
#             WhichIsTaken_list.append(bottle_list.pop(int(which.split('_')[1])))
#         except:
#             WhichIsTaken_list.append(bottle_list.pop(int(which.split('_')[1])-1))
#     WhichIsTaken = ','.join(str(x) for x in WhichIsTaken_list)

#     print(bottle_list)
#     print(int(WhichIsTaken_relative.split('_')[1]))
    
    #Get first
#     Objects_minXor_sort = sorted(Objects_minXor.items(), key=lambda x: x[1], reverse=True)
    WhichIsTaken_relative = Objects_minXor_sort[0][0]
    
#     WhichIsTaken = object_list.pop(WhichIsTaken_relative)
    WhichIsTaken = WhichIsTaken_relative
    
    return WhichIsTaken, object_list


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)