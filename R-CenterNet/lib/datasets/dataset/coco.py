from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCO(data.Dataset):
  num_classes = 47
  default_resolution = [512, 1024]
  original_res = [960, 1920]
  mean = np.array([0.52119324, 0.46639902, 0.41168393],
                  dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.19220782, 0.19731673, 0.20447321],
                  dtype=np.float32).reshape(1, 1, 3)


  def __init__(self, opt, split):
    super(COCO, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_{}2017.json').format(split)
    self.max_objs = 128
    self.class_name = [
        '__background__', 'airconditioner', 'backpack', 'bathtub', 'bed', 'board', 'book', 'bottle', 'bowl', 'bucket', 'cabinet', 'chair',
     'clock', 'clothes', 'computer', 'cup', 'cushion', 'door', 'extinguisher', 'fan', 'faucet', 'fireplace', 'heater',
     'keyboard', 'light', 'microwave', 'mirror', 'mouse', 'outlet', 'oven', 'paper extraction', 'person', 'phone',
     'picture', 'potted plant', 'refrigerator', 'shoes', 'shower head', 'sink', 'sofa', 'table', 'toilet', 'towel',
     'tv', 'vase', 'washer', 'window', 'wine glass']

    self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                       31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                       41, 42, 43, 44, 45, 46, 47]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float(x)

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          if self.opt.keep_res:
            theta = (bbox[0] - 0.5) * 2 * np.pi / self.original_res[1]
            phi = (bbox[1] - 0.5) * np.pi / self.original_res[0]
          else:
            bbox[0] = bbox[0] * self.default_resolution[1] / self.original_res[1]
            bbox[1] = bbox[1] * self.default_resolution[0] / self.original_res[0]
            theta = (bbox[0] - 0.5) * 2 * np.pi / self.default_resolution[1]
            phi = (bbox[1] - 0.5) * np.pi / self.default_resolution[0]
          score = bbox[5]
          bbox_out  = list(map(self._to_float,
                              [bbox[0], bbox[1], theta, phi, bbox[2], bbox[3], bbox[4]]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
