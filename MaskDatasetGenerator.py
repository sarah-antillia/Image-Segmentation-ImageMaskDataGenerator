# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ImageMaskDatasetGenerator.py
# 2023/08/20 to-arai

import os
import numpy as np
import cv2
import glob
import random
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback
from ConfigParser import ConfigParser

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"
MASK   = "mask"

class MaskDatasetGenerator:

  def __init__(self, config_file, dataset=TRAIN, seed=137):
    random.seed = seed
    config = ConfigParser(config_file)
    self.image_width    = config.get(MODEL, "image_width")
    self.image_height   = config.get(MODEL, "image_height")
    self.image_channels = config.get(MODEL, "image_channels")
    
    self.train_dataset  = [ config.get(TRAIN, "image_datapath"),
                            config.get(TRAIN, "mask_datapath")]
    
    self.eval_dataset   = [ config.get(EVAL, "image_datapath"),
                            config.get(EVAL, "mask_datapath")]

    self.binarize  = config.get(MASK, "binarize")
    self.threshold = config.get(MASK, "threshold")
    self.blur_mask = config.get(MASK, "blur")

    #Fixed blur_size
    self.blur_size = (3, 3)
    if not dataset in [TRAIN, EVAL]:
      raise Exception("Invalid dataset")
      
    image_datapath = None
    mask_datapath  = None
    self.batch_size  = config.get(TRAIN, "batch_size")
    
    [image_datapath, mask_datapath] = self.train_dataset
    if dataset == EVAL:
      [image_datapath, mask_datapath] = self.eval_dataset

    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)

    mask_files   = None
    if os.path.exists(mask_datapath):
      mask_files  = glob.glob(mask_datapath + "/*.jpg")
      mask_files += glob.glob(mask_datapath + "/*.png")
      mask_files += glob.glob(mask_datapath + "/*.bmp")
      mask_files += glob.glob(mask_datapath + "/*.tif")
      mask_files  = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")
    
    self.image_datapath = image_datapath
    self.mask_datapath  = mask_datapath
    
    self.image_files    = image_files
    self.mask_files     = mask_files
    

  def random_sampling(self, batch_size):
    #random.shuffle(self.image_files)
    images_sample = random.sample(self.image_files, batch_size)
    masks_sample  = []
    for image_file in images_sample:
      basename  = os.path.basename(image_file)
      mask_file = os.path.join(self.mask_datapath, basename)
      if os.path.exists(mask_file):
        masks_sample.append(mask_file)
      else:
        raise Exception("Not found " + mask_file)
    return (images_sample, masks_sample)
    
      
  def generate(self):
   
    while True:
      (image_files, mask_files) = self.random_sampling(self.batch_size)

      Y = []
      for n, mask_file in enumerate(mask_files):
        print(" {} mask_file {}".format(n, mask_file))
      
        mask  = cv2.imread(mask_file)
        mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask  = cv2.resize(mask, dsize= (self.image_height, self.image_width),   interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        if self.binarize:
          mask[mask< self.threshold] =   0  
          mask[mask>=self.threshold] = 255

        # Blur mask 
        if self.blur_mask:
          mask = cv2.blur(mask, self.blur_size)
  
        mask  = np.expand_dims(mask, axis=-1)
        _y = np.zeros((1, self.image_height, self.image_width, 1                ), dtype=np.bool)
        _y = mask
        Y.append(_y)

        yield Y
