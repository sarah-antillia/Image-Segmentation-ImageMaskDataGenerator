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

# ImageMaskAugmentor.py
# 2023/08/20 to-arai

import os
import sys
import numpy as np
import cv2
from ConfigParser import ConfigParser

MODEL     = "model"
GENERATOR = "generator"

class ImageMaskAugmentor:
  
  def __init__(self, config_file):
    self.config  = ConfigParser(config_file)
    self.rotation = self.config.get(GENERATOR, "rotation", dvalue=True)

    self.ANGLES   = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    self.ANGLES   = self.config.get(GENERATOR, "angles", dvalue=[60, 120, 180, 240, 300])
    self.debug    = self.config.get(GENERATOR, "debug", dvalue=True)
    self.W        = self.config.get(MODEL, "image_width")
    self.H        = self.config.get(MODEL, "image_height")

  # It applies  horizotanl and vertical flipping operations to image and mask repectively.
  def augment(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    """ 
    IMAGES: Python list
    MASKS:  Python list
    image:  OpenCV image
    mask:   OpenCV mask
    """
    hflip_image = self.horizontal_flip(image) 
    hflip_mask  = self.horizontal_flip(mask) 
    #print("--- hflp_mask shape {}".format(hflip_mask.shape))
    IMAGES.append(hflip_image )    
    MASKS.append( hflip_mask  )
    if self.debug:
      filepath = os.path.join(generated_images_dir, "hfliped_" + image_basename)
      cv2.imwrite(filepath, hflip_image)
      filepath = os.path.join(generated_masks_dir,  "hfliped_" + mask_basename)
      cv2.imwrite(filepath, hflip_mask)

    vflip_image = self.vertical_flip(image)
    vflip_mask  = self.vertical_flip(mask)
    #print("== vflip shape {}".format(vflip_mask.shape))
    IMAGES.append(vflip_image )    
    MASKS.append( vflip_mask  )
    if self.debug:
      filepath = os.path.join(generated_images_dir, "vfliped_" + image_basename)
      cv2.imwrite(filepath, vflip_image)
      filepath = os.path.join(generated_masks_dir,  "vfliped_" + mask_basename)
      cv2.imwrite(filepath, vflip_mask)

    if self.rotation:
       self.rotate(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

  def horizontal_flip(self, image): 
    image = image[:, ::-1, :]
    return image

  def vertical_flip(self, image):
    image = image[::-1, :, :]
    return image
  
  
  def rotate(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    for angle in self.ANGLES:      

      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H))
      rotated_mask  = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(self.W, self.H))
      IMAGES.append(rotated_image)
      rotated_mask  = np.expand_dims(rotated_mask, axis=-1) 
      #print("rotated_mask shape {}".format(rotated_mask.shape))
      MASKS.append(rotated_mask)

      if self.debug:
        filepath = os.path.join(generated_images_dir, "rotated_" + str(angle) + "_" + image_basename)
        cv2.imwrite(filepath, rotated_image)
        filepath = os.path.join(generated_masks_dir,  "rotated_" + str(angle) + "_" + mask_basename)
        cv2.imwrite(filepath, rotated_mask)
  