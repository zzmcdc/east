import mxnet as mx
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from mxnet.gluon.data import Dataset
from utils import check_and_validate_polys, crop_area, generate_rbox
import ipdb
import  traceback

class Textdetection(Dataset):
  def __init__(self, anno_path):
    super(Textdetection, self).__init__()
    with open(anno_path, 'r') as f:
      lines = f.readlines()
    self.anno = [item.strip().split(',') for item in lines]



  def __len__(self):
    return len(self.anno)

  def __getitem__(self, item):
    img_path = self.anno[item][0]

    # [x1,y1,x2,y2,x3,y3,x4,y4, mask]
    anno = self.anno[item][1:]
    img = cv2.imread(img_path)
    text_polys = []
    text_tags = []
    for i in range(0, len(anno), 9):
      line = anno[i:i + 9]
      x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
      text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
      tag = True if line[8] == 1 else False
      text_tags.append(tag)

    h, w, c = img.shape
    text_polys, text_tags = check_and_validate_polys(np.array(text_polys, dtype=np.float32),
                                                     np.array(text_tags, dtype=np.bool), (h, w))
    return img, text_polys, text_tags

class EASTDefaultTrainTransform(object):
  def __init__(self, param):
    self.param = param

  def __call__(self, img, text_polys, text_tags):
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    rd_scale = np.random.choice(random_scale)
    # img = random_color_distort(img)
    im = cv2.resize(img, None, fx=rd_scale, fy=rd_scale)
    if np.random.rand() < self.param['background_ratio']:
      im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True, param=self.param)
      new_h, new_w, _ = im.shape
      max_h_h_i = np.max([new_h, new_w, self.param['height'], self.param['width']])
      im_padded = np.zeros((max_h_h_i, max_h_h_i, 3), dtype=np.uint8)
      im_padded[:new_h, :new_w, :] = im.copy()
      im = cv2.resize(im_padded, dsize=(self.param['width'], self.param['height']))
      score_map = np.zeros((self.param['height'], self.param['width']))
      geo_map = np.zeros((5, self.param['height'], self.param['width']))
      training_mask = np.ones((self.param['height'], self.param['width']))
    else:
      im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False, param=self.param)
      new_h, new_w, _ = im.shape
      max_h_h_i = np.max([new_h, new_w, self.param['height'], self.param['width']])
      im_padded = np.zeros((max_h_h_i, max_h_h_i, 3), dtype=np.uint8)
      im_padded[:new_h, :new_w, :] = im.copy()
      im = im_padded

      new_h, new_w, _ = im.shape
      resize_h = self.param['height']
      resize_w = self.param['width']
      im = cv2.resize(im, dsize=(resize_w, resize_h))
      resize_ratio_3_x = resize_w / float(new_w)
      resize_ratio_3_y = resize_h / float(new_h)
      text_polys[:, :, 0] *= resize_ratio_3_x
      text_polys[:, :, 1] *= resize_ratio_3_y

      new_h, new_w, _ = im.shape
      score_map, geo_map, training_mask = generate_rbox(new_h, new_w, text_polys, text_tags, param=self.param)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = mx.nd.array(im)
    image = mx.nd.image.to_tensor(image)
    image = mx.nd.image.normalize(image, mean=mean, std=std)

    score_maps = mx.nd.array(score_map[np.newaxis,::4, ::4])
    geo_maps = mx.nd.array(geo_map[:,::4, ::4])
    geo_maps[0:4,:,:] = geo_maps[0:4,:,:]/self.param['height']
    training_masks = mx.nd.array(training_mask[np.newaxis,::4, ::4])
    # print(image.shape, score_maps.shape, geo_maps.shape, training_masks.shape)

    return image, score_maps, geo_maps, training_masks

def load_image(img, data_shape):
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  new_h, new_w, _ = img.shape
  max_h_h_i = np.max([new_h, new_w, data_shape[0], data_shape[1]])
  im_padded = np.zeros((max_h_h_i, max_h_h_i, 3), dtype=np.uint8)
  im_padded[:new_h, :new_w, :] = img.copy()
  im_padded = cv2.resize(im_padded,(data_shape[0],data_shape[1]))
  img_out = im_padded
  img = mx.nd.array(im_padded)
  img = mx.nd.image.to_tensor(img)
  img = mx.nd.image.normalize(img, mean=mean, std=std)

  return img_out, img


class EASTDefaultValTransform(object):
  def __init__(self, data_shape):
    self.data_shape = data_shape
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)

  def __call__(self, img, text_polys, text_tags):
    # ipdb.set_trace()
    new_h, new_w, _ = img.shape

    max_h_h_i = np.max([new_h, new_w, self.data_shape[0], self.data_shape[1]])
    im_padded = np.zeros((max_h_h_i, max_h_h_i, 3), dtype=np.uint8)

    im_padded[:new_h, :new_w, :] = img.copy()
    img = mx.nd.array(im_padded)
    img = mx.nd.image.resize(img, size=(self.data_shape))
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)

    resize_ratio_3_x = self.data_shape[0] / float(max_h_h_i)
    resize_ratio_3_y = self.data_shape[1] / float(max_h_h_i)
    text_polys[:, :, 0] *= resize_ratio_3_x
    text_polys[:, :, 1] *= resize_ratio_3_y
    return img, text_polys.reshape(-1,8)
