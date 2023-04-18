# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The MIT License (MIT)
# Copyright (c) 2023 Thomas DUMAZERT
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ----------------------------------------------
# | script: bcc_utilities.py                   |
# | author: Thomas DUMAZERT                    |
# | creation: 02/13/2023                       |
# | last modified: 03/21/2023                  |
# ----------------------------------------------

# This module is intended to provided utility functions to analyze and handle
# the data of the 'Blood cells classification' project.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# External libraries

import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from PIL import Image
import re
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import custom scripts
import sys
sys.path.append("..")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants

BLOOD_CELL_ENCODER = {
    'artifact':'art',
    'basophil': 'bas',
    'blast_cell': 'myb', # Previously 'blt'
    'eosinophil': 'eos',
    'erythrocyte': 'ery',
    'immature_granulocyte': 'img',
    'lymphocyte': 'lym',
    'metamyelocyte': 'img', # Previously 'mmc'
    'monocyte': 'mon',
    'myeloblast': 'myb',
    'myelocyte': 'img', # Previously 'myc'
    'neutrophil': 'neu',
    'platelet': 'plt',
    'promyelocyte': 'img', # Previously 'pyc'
    'unknown_precursor': 'ukp',
}

# Barcelona and Raabin blood cell transform dictionnary
BR = {
    'arifact': 'artifact',
    'band neutrophils': 'neutrophil',
    'basophil': 'basophil', 
    'blast, no lineage spec': 'myeloblast', 
    'eosinophil': 'eosinophil',
    'eosinophils': 'eosinophil',
    'erythroblast': 'erythrocyte',
    'giant thrombocyte': 'platelet',
    'ig': 'immature_granulocyte', 
    'lymphocite': 'lymphocyte',
    'lymphocyte': 'lymphocyte',
    'lymphocyte, variant': 'lymphocyte',
    'metamyelocyte': 'metamyelocyte',
    'monocyte': 'monocyte', 
    'myelocyte': 'myelocyte',
    'neutrophil': 'neutrophil', 
    'plasma cells': 'lymphocyte',
    'platelet': 'platelet', 
    'prolymphocyte': 'lymphocyte',
    'promonocyte': 'monocyte',
    'promyelocyte': 'promyelocyte',
    'segmented neutrophils': 'neutrophil', 
    'smudge cells': 'artifact',
    'thrombocyte aggregation': 'platelet', 
    'unidentified': 'artifact', 
    'young unidentified': 'unknown_precursor'
}

# Munich and Kaggle blood cell transform dictionnary
MK = {
    'art': 'artifact',
    'ba': 'basophil',
    'bas': 'basophil',
    'bl': 'myeloblast',
    'bne': 'neutrophil',
    'eo': 'eosinophil',
    'eob': 'erythrocyte',
    'eos': 'eosinophil',
    'erb': 'erythrocyte',
    'erc': 'platelet',
    'gt': 'platelet',
    'ksc': 'artifact',
    'ly': 'lymphocyte',
    'lya': 'lymphocyte',
    'lyt': 'lymphocyte',
    'mmy': 'metamyelocyte',
    'mmz': 'metamyelocyte',
    'mo': 'monocyte',
    'mob': 'monocyte',
    'mon': 'monocyte',
    'my': 'myelocyte',
    'myb': 'myelocyte',
    'myo': 'myelocyte',
    'ngb': 'neutrophil',
    'ngs': 'neutrophil',
    'pc': 'lymphocyte',
    'ply': 'lymphocyte',
    'pmb': 'promyelocyte',
    'pmo': 'promyelocyte',
    'pmo': 'monocyte',
    'pmy': 'promyelocyte',
    'smu': 'artifact',
    'sne': 'neutrophil',
    'vly': 'lymphocyte',
}

IMG_EXTS = ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', 
    '.bmp', '.svg', '.webp', '.psd', '.ico', '.raw', '.heic', '.nef']

SOURCES = ['barcelone', 'kaggle', 'munich', 'raabin']

# Dictionnary linking sources with their transform dictionnary
TRANSFORMS = {
    'barcelone': BR,
    'kaggle': MK,
    'munich': MK,
    'raabin': BR,
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Class

class ImageAugmentation:
    def __init__(self, path, file_name, rotation=None, hflip=False, vflip=False, contrast=None, brightness=None, random_state=None, random_generator=None):
        """
        Initailize instance.
        Parameters:
            - 'path': string. Path to the image ;
            - 'file_name': string. Image file name ;
            - 'rotation': int or array-like of ints. Number of 90 degrees 
                          rotation to choose from. Default to None ;
            - 'hflip': boolean. Flag to authorize horizontal flip. Default to 
                       None ;
            - 'vflip': boolean. Flag to authorize vertical flip. Default to 
                       None ;
            - 'contrast': float or array-like of floats. Boundaries of the 
                          contrast modification. Default to None ;
            - 'brightness': int or array-like of ints. Boundaries of the 
                            brightness modification. Default to None ;
        - 'random_state': optionnal, int. A random seed to initialize a 
                          pseudo-random sequence of numbers. Default to None0 ;
        - 'random_generator': optionnal, pseudo-random number generator 
                              object. Use to continue a pseudo-random sequence 
                              already initialized. Default to None.
        Return: None
        """
        
        self.path = path
        self.name = file_name
        if isinstance(rotation, int): self.rotation = range(rotation+1)
        else: self.rotation = rotation
        self.hflip = hflip
        self.vflip = vflip
        self.contrast = contrast
        self.brightness = brightness

        # Initialize a pseudo-random sequence
        if not random_generator: self.random_generator = random.Random(random_state)
        else: self.random_generator = random_generator

        self.augmentations = [self.rotate, self.flip, self.change_colors]

    def rotate(self, image):
        """
        Randomly rotate the image.
        Parameter: 
            - 'image': numpy array. The image to be processed.
        Return: the processed image.
        """

        if not self.rotation: return image

        h, w, _ = image.shape

        # Choose a random angle
        angle = self.random_generator.choice(self.rotation) * 90

        # Rotation matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)

        # Rotate
        image = cv2.warpAffine(image,M,(w,h))

        return image

    def flip(self, image):
        """
        Randomly flip the image.
        Parameter: 
            - 'image': numpy array. The image to be processed.
        Return: the processed image.
        """

        if self.hflip or self.vflip:
            if self.hflip:
                if self.random_generator.randint(0, 1):
                    image = cv2.flip(image, flipCode=1)
            if self.vflip:
                if self.random_generator.randint(0, 1):
                    image = cv2.flip(image, flipCode=0)
            
        return image

    def change_colors(self, image):
        """
        Randomly change the image brightness.
        Parameter: 
            - 'image': numpy array. The image to be processed.
        Return: the processed image.
        """

        # Contrast and brightness parameters
        alpha, beta = 1.0, 0

        if self.contrast:
            low_contrast, high_contrast = 0.0, 0.0
            try:
                low_contrast, high_contrast = self.contrast
            except:
                high_contrast = self.contrast

            # Range up to high_contrast+0.00001 because np.random.uniform 
            # upper boundary is exclusive
            alpha = self.random_generator.uniform(low_contrast, high_contrast+0.00001)

            # Bound alpha parameter to 3.0
            alpha = 3.0 if alpha > 3.0 else alpha
        
        if self.brightness:
            low_brightness, high_brightness = 0, 0
            try:
                low_brightness, high_brightness = self.brightness
            except:
                high_brightness = self.brightness

            beta = self.random_generator.randint(low_brightness, high_brightness)

        # Apply parameters
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image

    def image_augment(self, save_path, num_augments=1):
        """
        Create new images with image augmentations.
        Parameter:
            - 'save_path': string. The path to store the new image ;
            - 'num_augments': optionnal, int. The number of augmentation to 
                              proceed. Default to 1.
        Return: the path to tthe new image
        """

        img_aug = cv2.imread(f'{self.path}/{self.name}')
        augs = self.random_generator.sample(self.augmentations, k=self.random_generator.randint(1, len(self.augmentations)))

        for i in range(num_augments):
            for aug in augs:
                img_aug = aug(img_aug)
            img_name = f'{self.name.split(".")[0]}_aug_{i+1:03d}.jpg'
            cv2.imwrite(f'{save_path}/{img_name}', img_aug)

        return save_path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def file_is_a(file, exts):
    """
    Wether the provided file is a type of of file whose extensions are 
    provided.
    Parameters:
        - 'file': a string containing the name of the file with its extension ;
        - 'exts': a list of the file extensions to consider.
    Return: a boolean.
    """

    ext = file.split('.')[-1]
    return f'.{ext}' in exts

def split_path(path):
    """
    Split a path into its components.
    Parameter:
        - 'path': a string containing a path.
    Return: a list containing the directories and eventually the file of the 
            provided path.
    """

    splitted_path = path.split('\\')
    if len(splitted_path) == 1:
        splitted_path = path.split('/')
    return splitted_path

def encode_blood_cell(blood_cell, transform_dict, encoder=BLOOD_CELL_ENCODER):
    """
    Encode a blood cell name according to the encoder dictionnary provided.
    Parameters:
        - 'blood_cell': a string containing the name of a blood cell ;
        - 'transform_dict': a dictionnary whose keys are blood cells
                            names in the original dataset and values 
                            are new names encoded blood cells names ;
        - 'encoder': a dictionnary mapping blood cells names and their
                     encoded names.
    Return: the encoded blood cell name of the transformed name.
    """

    if blood_cell not in transform_dict.keys():
        return 'xxx'
    else:
        return encoder[transform_dict[blood_cell]]
    
def get_source(path, sources=SOURCES):
    """
    Extracts the source of the dataset from a path.
    Parameters:
        - 'path': a string containing the path to analyze ;
        - 'sources': optionnal. A list containing the potential sources.
    Return: None or the name of the source dataset.
    """

    path = split_path(path)
    for dir in path:
        if dir in sources:
            return dir
    return None

def get_complete_blood_cell(encoded_blood_cell, encoder=BLOOD_CELL_ENCODER):
    """
    Returns the complete name from an encoded one.
    Parameter:
        - 'encoded_blood_cell': a string containing the encoded name ;
        - 'encoder': a dictionnary mapping blood cells names and their
                     encoded names.
    Return: the complete name of the blood cell
    """

    return list(encoder.keys())[list(encoder.values()).index(encoded_blood_cell)]

def count_image(root):
    """

    """

    img_count = 0

    for _, _, files_list in os.walk(root):
        for file in files_list:
            if file_is_a(file, IMG_EXTS): img_count += 1

    return img_count

def organize_data(source, target, transforms_dict=TRANSFORMS, adjust_size=False, target_size=None, size_adjust_dir=None, resize=None, exclude=None):
    """
    Sort and rename the files from the source directory and put them into the
    target directory.
    Parameters:
        - 'source': a string containing the path (relative or absolute) of the 
                    source directory ;
        - 'target': a string containing the path (relative or absolute) of the 
                    target directory ;
        - 'transforms_dict': a dictionnary mapping the sources of the datasets 
                             with their transform dictionnary ;
        - 'adjust_size': a boolean flagging the need to adjusts images size ;
        - 'target_size': an int or a tuple containing the targeted (height, 
                         width) of the adjusted images. Should be provided if 
                         'adjust_size' is set to True. If int width = height ;
        - 'size_adjust_dir': string. The path where to store the resized 
                             images. If None, the resized images are stored in 
                             the target directory instead of the original 
                             ones. If a path is indicated both original images 
                             and resized images are stored in their respective 
                             directories ;
        - 'resize': a dictionnary containing the image to resize instead of crop.
                    Maps the sources with a list of encoded blood cell to crop ;
        - 'exclude': a list containing the categories to exclude from the 
                     dataset.
    Return: the number of images treated.
    """

    if not Path(target).exists(): os.mkdir(target)
    
    target_base = f'{target}/base'
    if not Path(target_base).exists(): os.mkdir(target_base)

    if adjust_size and not target_size:
        raise AttributeError("'adjust_size' set to true without providing 'target_size'")

    if type(target_size) == int: target_size = (target_size, target_size)

    img_counter = 1
    
    for current_dir, _, files_list in os.walk(source):
        ds_source = get_source(current_dir)
        
        patient_search = re.search(r'Patient_\d+', current_dir)
        if patient_search:
            patient_num = f'{int(patient_search.group(0).split("_")[-1]):03d}'
        else:
            patient_num = 'xxx'
        
        for file in files_list:
            blood_cell = split_path(current_dir)[-1].lower()

            if file_is_a(file, IMG_EXTS):
                # Images in kaggle dataset may be miss sorted
                # We have to get file name prefix to ensure proper sort
                if ds_source == 'kaggle':
                    blood_cell = file.split('_')[0].lower()

                encoded_blood_cell = encode_blood_cell(blood_cell, transforms_dict[ds_source])

                if encoded_blood_cell not in exclude:
                    if ds_source == 'kaggle' and current_dir.find('Unsigned slides') == -1:
                        encoded_blood_cell = encoded_blood_cell.upper()
                
                    img_name = f'{encoded_blood_cell}_{ds_source[:3]}_{patient_num}_{img_counter:06d}.jpg'
                    img_counter += 1
                    
                    path = f'{current_dir}/{file}'
                    
                    corrupted = False
                    try:
                        Image.open(path).load()
                    except Exception as e:
                        corrupted = True

                    if not corrupted:
                        cell_dir = 'unlabeled' if encoded_blood_cell.lower() == 'xxx' else encoded_blood_cell.lower()
                        
                        target_dir = f'{target_base}/{cell_dir}'

                        if not Path(target_dir).exists(): os.mkdir(target_dir)

                        # Create adjusted images directory if needed
                        
                        if size_adjust_dir:
                            if not Path(size_adjust_dir).exists(): os.mkdir(size_adjust_dir)

                            adjust_base_dir = f'{size_adjust_dir}/base'
                            if not Path(adjust_base_dir).exists(): os.mkdir(adjust_base_dir)

                            adjust_target_dir = f'{adjust_base_dir}/{cell_dir}'
                            if not Path(adjust_target_dir).exists(): os.mkdir(adjust_target_dir)

                        target_file = f'{target_dir}/{img_name}'
                        origin_file = f'{current_dir}/{file}'
                        
                        if adjust_size:
                            img = cv2.imread(str(origin_file), cv2.IMREAD_COLOR)
                            crop = resize and (ds_source[:3] in resize.keys()) and (encoded_blood_cell in resize[ds_source[:3]])
                            img = adapt_size(img, target_size, crop=crop)

                            if size_adjust_dir:
                                adjust_target_file = f'{adjust_target_dir}/{img_name}'

                                cv2.imwrite(adjust_target_file, img)

                                if not Path(target_file).exists():
                                    shutil.copy2(origin_file, target_file)
                            else:
                                cv2.imwrite(target_file, img)
                        else:
                            if not target_file.exists():
                                shutil.copy2(origin_file, target_file)
                    
                    print(img_name)

    return img_counter

def get_color_hist(src, hist_size=256, hist_range=(0, 256), accumulate=False):
    """
    Extracts BGR pixels distribution from a color image.
    Parameters:
        - 'src': a numpy array of shape (h, w, 3) ;
        - 'hist_size':  ;
        - 'hist_range':  ;
        - 'accumulate': .
    Return: a numpy array of shape ('hist_size', 3) containing the pixels 
            values distributions for BGR channels.
    """

    b_hist = cv2.calcHist(src, [0], None, [hist_size], hist_range, accumulate=accumulate).reshape((hist_size,))
    g_hist = cv2.calcHist(src, [1], None, [hist_size], hist_range, accumulate=accumulate).reshape((hist_size,))
    r_hist = cv2.calcHist(src, [2], None, [hist_size], hist_range, accumulate=accumulate).reshape((hist_size,))

    return np.array([b_hist, g_hist, r_hist])

def get_gray_hist(src, hist_size=256, hist_range=(0, 256), accumulate=False):
    """
    Extracts pixels distribution from a grayscale image.
    Parameters:
        - 'src': a numpy array of shape (h, w) ;
        - 'hist_size':  ;
        - 'hist_range':  ;
        - 'accumulate': .
    Return: a numpy array of shape ('hist_size',) containing the pixels values 
            distributions.
    """

    hist = cv2.calcHist(src, [0], None, [hist_size], hist_range, accumulate=accumulate).reshape((hist_size,))

    return hist

def get_hist(src, hist_size=256, hist_range=(0, 256), accumulate=False):
    """
    Extracts pixels distribution from an image.
    Parameters:
        - 'src': a numpy array of shape (h, w, 3) or (h, w) ;
        - 'hist_size':  ;
        - 'hist_range':  ;
        - 'accumulate': .
    Return: a numpy array of shape ('hist_size', 3) or ('hist_size',) 
            containing the pixels values distributions.
    """

    if len(src[0].shape) == 3:
        return get_color_hist(src, hist_size, hist_range, accumulate)
    else:
        return get_gray_hist(src, hist_size, hist_range, accumulate)

def analyze_pixel_distribution(root):
    """
    Extracts pixels values grayscale and BGR distributions from images in the 
    root directory and its sub directories.
    Parameter:
        - 'root': a string containing the path (relative or absolute) to the 
                  root directory to analyze.
    Return: a dictionnary containing, for each image found in the root 
            directory and its sub directories, the gray, red, green and blue 
            pixels values distributions.
    """

    distributions = {
        'img_id': [],
        'blood_cell': [],
        'source': [],
        'healthy': [],
        'gray_distribution': [],
        'red_distribution': [],
        'green_distribution': [],
        'blue_distribution': [],
    }

    dir_list = os.listdir(root)

    for dir in dir_list:
        dir_path = Path(f'{root}/{dir}')
        if os.path.isdir(dir_path) and dir != 'mean_images':
            files_list = os.listdir(dir_path)
            for file in files_list:
                if file_is_a(file, IMG_EXTS):
                    img_id = file
                    blood_cell, source, _, _ = img_id.split('_')
                    healthy = blood_cell.isupper()
                    blood_cell = blood_cell.lower()

                    file_path = f'{root}/{dir}/{file}'
                    img_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                    color_hist = get_hist([img_color])
                    gray_hist = get_hist([img_gray])

                    distributions['img_id'].append(img_id)
                    distributions['blood_cell'].append(blood_cell)
                    distributions['source'].append(source)
                    distributions['healthy'].append(healthy)
                    distributions['gray_distribution'].append(gray_hist)
                    distributions['red_distribution'].append(color_hist[2, :])
                    distributions['green_distribution'].append(color_hist[1, :])
                    distributions['blue_distribution'].append(color_hist[0, :])
    
    return distributions

def get_img_path(root, img_name):
    """
    Rebuild the path to reach an image from its name and the root directory 
    containing the sub directories stocking the images.
    Parameters:
        - 'root': a string containing the path (relative or absolute) to the 
                  root directory ;
        - 'img_name': a string containing the name of the image from which to 
                      rebuild the path.
    Return: the path to reach the image.
    """

    img_dir = img_name.split('_')[0].lower()
    if img_dir == 'xxx': img_dir = 'unlabeled'
    return f'{root}/{img_dir}/{img_name}'

def adapt_gray_size(img, target_size, crop=True, centered=True, resize_center=None):
    """
    Adjust the size of a grayscale image to the targeted size and targeted 
    location.
    Parameters:
        - 'img': an array containing the pixels values of the image ;
        - 'target_size': a tuple containing the targeted (height, width) of the 
                         image after the size adjustment ;
        - 'crop': optionnal. A boolean indicating if the size adjustment must be
                  done through cropping or resizing. Default to True ;
        - 'centered': optionnal. A boolean indicating if the size adjustement 
                      must be centered on the center of the image or not. 
                      'resize_center' must be provided if set to False. Default 
                      to True ;
        - 'resize_center': optionnal. A tuple containing the target (x, y) 
                           coordinates of the non centerd size adjustment. 
                           Must be provided if 'centered' is set to False.
                           Default to None.
    Return: an array of shape 'target_size' containing the adjusted image.
    """

    if not centered and not resize_center:
        raise AttributeError(
            "'centered' set to False without providind 'resize_center' coordinates"
        )
    
    adjusted_img = None

    img_h, img_w = img.shape
    tgt_h, tgt_w = target_size

    if crop:
        if centered:
            if img_h >= tgt_h:
                y_start = int(np.ceil((img_h - tgt_h) / 2))
                y_end = y_start + tgt_h

                adjusted_img = img[y_start:y_end, :]
            else:
                y_start = int(np.ceil((tgt_h - img_h) / 2))
                y_end = y_start + img_h

                adjusted_img = np.ones((tgt_h, img_w)) * 255
                adjusted_img[y_start:y_end, :] = img

            if img_w >= tgt_w:
                x_start = int(np.ceil((img_w - tgt_w) / 2))
                x_end = x_start + tgt_w

                adjusted_img = adjusted_img[:, x_start:x_end]
            else:
                x_start = int(np.ceil((tgt_w - img_w) / 2))
                x_end = x_start + img_w

                buffer_img = np.ones((adjusted_img.shape[0], tgt_w)) * 255
                buffer_img[:, x_start:x_end] = adjusted_img

                adjusted_img = buffer_img
        else:
            raise NotImplementedError(
                "Non centered size modification is not yet implemented"
            )
    else:
        adjusted_img = cv2.resize(img, target_size)
    
    return adjusted_img

def adapt_color_size(img, target_size, crop=True, centered=True, resize_center=None):
    """
    Adjust the size of a color image to the targeted size and targeted 
    location.
    Parameters:
        - 'img': an array containing the pixels values of the image ;
        - 'target_size': a tuple containing the targeted (height, width) of the 
                         image after the size adjustment ;
        - 'crop': optionnal. A boolean indicating if the size adjustment must be
                  done through cropping or resizing. Default to True ;
        - 'centered': optionnal. A boolean indicating if the size adjustement 
                      must be centered on the center of the image or not. 
                      'resize_center' must be provided if set to False. Default 
                      to True ;
        - 'resize_center': optionnal. A tuple containing the target (x, y) 
                           coordinates of the non centerd size adjustment. 
                           Must be provided if 'centered' is set to False.
                           Default to None.
    Return: an array of shape 'target_size' containing the adjusted image.
    """

    if not centered and not resize_center:
        raise AttributeError(
            "'centered' set to false without providind 'resize_center' coordinates"
        )

    adjusted_img = None

    img_h, img_w = img.shape[:2]
    tgt_h, tgt_w = target_size

    if crop:
        if centered:
            if img_h >= tgt_h:
                y_start = int(np.ceil((img_h - tgt_h) / 2))
                y_end = y_start + tgt_h

                adjusted_img = img[y_start:y_end, :, :]
            else:
                y_start = int(np.ceil((tgt_h - img_h) / 2))
                y_end = y_start + img_h

                adjusted_img = np.ones((tgt_h, img_w, 3)) * 255
                adjusted_img[y_start:y_end, :, :] = img

            if img_w >= tgt_w:
                x_start = int(np.ceil((img_w - tgt_w) / 2))
                x_end = x_start + tgt_w

                adjusted_img = adjusted_img[:, x_start:x_end, :]
            else:
                x_start = int(np.ceil((tgt_w - img_w) / 2))
                x_end = x_start + img_w

                buffer_img = np.ones((adjusted_img.shape[0], tgt_w, 3)) * 255
                buffer_img[:, x_start:x_end, :] = adjusted_img

                adjusted_img = buffer_img
        else:
            raise NotImplementedError(
                "Non centered size modification is not yet implemented"
            )
    else:
        adjusted_img = cv2.resize(img, target_size)
    
    return adjusted_img.astype(np.int16)

def adapt_size(img, target_size, crop=True, centered=True, resize_center=None):
    """
    Adjust the size of an image to the targeted size and targeted 
    location.
    Parameters:
        - 'img': an array containing the pixels values of the image ;
        - 'target_size': a tuple containing the targeted (height, width) of the 
                         image after the size adjustment ;
        - 'crop': optionnal. A boolean indicating if the size adjustment must be
                  done through cropping or resizing. Default to True ;
        - 'centered': optionnal. A boolean indicating if the size adjustement 
                      must be centered on the center of the image or not. 
                      'resize_center' must be provided if set to False. Default 
                      to True ;
        - 'resize_center': optionnal. A tuple containing the target (x, y) 
                           coordinates of the non centerd size adjustment. 
                           Must be provided if 'centered' is set to False.
                           Default to None.
    Return: an array of shape 'target_size' containing the adjusted image.
    """

    if not centered and not resize_center:
        raise AttributeError(
            "'centered' set to false without providind 'resize_center' coordinates"
        )

    if len(img.shape) == 3: return adapt_color_size(img, target_size, crop, centered, resize_center)
    else: return adapt_gray_size(img, target_size, crop, centered, resize_center)

def mean_image(path_list):
    """
    Satcks images and compute the average pixels values.
    Parameter:
        - 'path_list': a list of path pointing to the images to average.
    Return: an array containing the mean values of the pixels of the images
            from the 'path_list'.
    """

    max_width = 0
    max_height = 0
    
    for path in path_list:
        with Image.open(path) as img:
            width, height = img.size
        if width > max_width: max_width = width
        if height > max_height: max_height = height
    
    if max_width > max_height: max_height = max_width
    else: max_width = max_height
    
    mean_image = adapt_size(cv2.imread(path_list[0], cv2.IMREAD_GRAYSCALE), (max_width, max_height)).astype(np.int64)

    for i in range(1, len(path_list)):
        mean_image += adapt_size(cv2.imread(path_list[i], cv2.IMREAD_GRAYSCALE), (max_width, max_height)).astype(np.int64)

    return np.round(mean_image / len(path_list)).astype(np.int16)

def str_to_list(array_string, dtype=float):
    """
    Convert a string containing literal expression of a python list to a 
    python list.
    Parameters:
        - 'array_string': a string containing literal expression of a python 
                          list ;
        - 'dtype': the target data type of items in the list.
    Return: a list.
    """

    list_string = array_string.replace('[', '').replace(']', '').split()
    return [dtype(num) for num in list_string]

def srs_to_array(srs):
    """
    Convert a pandas Series to an array.
    Parameter:
        - 'srs': a pandas Series.
    Return: an array containing the values of the original Series.
    """

    return np.array([r for r in srs.values])

def cumsum(l):
    """
    Compute cumulative sum of a list.
    Parameter:
        - 'l': a list summables ;
    Return: a list containing the cumulated values of the original list.
    """

    csl = [l[0]]

    for i in range(1, len(l)):
        csl.append(csl[i-1] + l[i])

    return csl

def bbox(img):
    """
    Found bouding box of a white form in a binary image.
    Parameter:
        - 'img': an array containing pixels values of a binary image.
    Return: 'ymin', 'ymax', 'xmin' and 'xmax' where (xmin, ymin) is the upper 
            left corner of the bounding box and (xmax, ymax) is the lower 
            right corner of the bounding box.
    """

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return ymin, ymax, xmin, xmax

def image_data_augmentation(source_dir, target_dir, num_images_per_cat=1500, random_state=None, random_generator=None):
    """
    Apply data auggmentations to exceed to targeted number of images.
    Parameters:
        - 'source_dir': string. The directory to select images from ;
        - 'target_dir': string. The directory to save the new images ;
        - 'num_images_per_cat': optionnal, int stricly greater than zero. 
                                The targeted number of images per category. 
                                Default to 1500 ;
        - 'random_state': optionnal, int. A random seed to initialize a 
                          pseudo-random sequence of numbers. Default to None0 ;
        - 'random_generator': optionnal, pseudo-random number generator 
                              object. Use to continue a pseudo-random sequence 
                              already initialized. Default to None.
    Return: the path to the augmented images directory.
    """

    if num_images_per_cat <= 0:
        raise AttributeError(f'num_images_per_cat can\'t be lower or equal to zero, here {num_images_per_cat}')

    img_list = os.listdir(source_dir)

    num_images = len(img_list)

    # Compute number of augmentation to apply
    n_augmentation = int(np.ceil((num_images_per_cat - num_images) / num_images))

    # Add one to ensure a the application of at least one augmentation.
    n_augmentation += 1 if n_augmentation == 0 else 0

    # Initialize a pseudo-random sequence
    if not random_generator:
        random_generator = random.Random(random_state)

    for img in img_list:
        target_file = Path(f'{target_dir}/{img}')
        origin_file = Path(f'{source_dir}/{img}')
        shutil.copy2(origin_file, target_file)

        aug_img = ImageAugmentation(
            path = source_dir,
            file_name = img,
            rotation = 360,
            hflip = True,
            vflip = True,
            contrast = (0.5, 2.0),
            brightness = (0, 50),
            random_generator = random_generator
        )

        aug_img.image_augment(target_dir, num_augments=n_augmentation)

    return target_dir

def select_random_sample(source_dir, target_dir, cat, num_images_per_cat, test_size=0.1, val_size=0.2, random_state=None, random_generator=None):
    """
    Select a random sample from the source directory and copy the images to 
    the targeted directory.
    If the source directory is the same as the targeted directory, delete the 
    images not selected.
    Parameters:
        - 'source_dir': string. The directory to selecting images from ;
        - 'target_dir': string. The directory where to save the sets ;
        - 'cat': string. The name of the category ;
        - 'num_images_per_cat': int. The targeted number of images per 
                                category. If zero or lower, split every 
                                images in the source directory ;
        - 'test_size': float. Strictly between 0.0 and 1.0. The size of the 
                       test set to create. Default to 0.1  ;
        - 'val_size': float. Strictly between 0.0 and 1.0. The size of the 
                      validation set to create. Default to 0.2  ;
        - 'random_state': optionnal, int. A random seed to initialize a 
                          pseudo-random sequence of numbers. Default to None0 ;
        - 'random_generator': optionnal, pseudo-random number generator 
                              object. Use to continue a pseudo-random sequence 
                              already initialized. Default to None.
    Return: the path to the selected images directory.
    """

    # Create the folder to store the sets
    train_dir = Path(f'{target_dir}/train/{cat}')
    test_dir = Path(f'{target_dir}/test/{cat}')
    val_dir = Path(f'{target_dir}/val/{cat}')

    # Check whether the directories exist or not and create if needed
    if not train_dir.exists(): os.mkdir(train_dir)
    if not test_dir.exists(): os.mkdir(test_dir)
    if not val_dir.exists(): os.mkdir(val_dir)

    img_list = os.listdir(source_dir)

    num_images = len(img_list)

    # Initialize a pseudo-random sequence
    if not random_generator:
        random_generator = random.Random(random_state)
    
    if num_images_per_cat <= 0: num_images_per_cat = num_images
    
    print(f'sampling pop = {num_images}', f'sampled pop = {num_images_per_cat}', sep='\t', end='\t')

    selection = random_generator.sample(range(num_images), num_images_per_cat)

    num_test = int(np.floor(num_images_per_cat * test_size))
    num_val = int(np.floor(num_images_per_cat * val_size))
    num_train = num_images_per_cat - (num_test + num_val)

    print(f'train = {num_train}', f'val = {num_val}', f'test = {num_test}', sep='\t', end='\t')

    # Create the test selection
    if test_size > 0.0: test_selection = random_generator.sample(selection, num_test)
    else: test_selection = []

    # Remove images selected to be in the test set from the initial selection
    selection = [idx for idx in selection if idx not in test_selection]

    # Create the validation selection
    if val_size > 0.0: val_selection = random_generator.sample(selection, num_val)
    else: val_selection = []

    # Remove images selected to be in the validation set from the initial selection
    train_selection = [idx for idx in selection if idx not in val_selection]

    for idx, img_name in enumerate(img_list):
        origin_file = Path(f'{source_dir}/{img_name}')
        
        if source_dir == target_dir:
            if idx in train_selection: shutil.move(origin_file, Path(f'{train_dir}/{img_name}'))
            elif idx in test_selection: shutil.move(origin_file, Path(f'{test_dir}/{img_name}'))
            elif idx in val_selection: shutil.move(origin_file, Path(f'{val_dir}/{img_name}'))
            else: os.remove(origin_file)
        else:
            if idx in train_selection: shutil.copy2(origin_file, Path(f'{train_dir}/{img_name}'))
            elif idx in test_selection: shutil.copy2(origin_file, Path(f'{test_dir}/{img_name}'))
            elif idx in val_selection: shutil.copy2(origin_file, Path(f'{val_dir}/{img_name}'))


def create_dataset(root_dir, num_images_per_cat=1500, test_size=0.1, val_size=0.2, random_state=None, random_generator=None):
    """
    Create a dataset by augmented (if needed) and selecting the images from 
    categories. The new directory will have the same organization as the 
    original one.
    If the number of images is lower or equal to the targeted number of 
    images, data augmentation is applied.
    A random sampling is then applied to selected the targeted number of 
    images.
    Parameters:
        - 'root_dir': string. The directory that store the directory to find 
                        the images. Directories created during dataset 
                        creation process (augmented images and sets) are 
                        created in this directory ;
        - 'num_images_per_cat': optionnal, int. The targeted number of images 
                                per category. If zero or lower, doesn't 
                                balanced categories. Default to 1500 ;
        - 'test_size': float. Strictly between 0.0 and 1.0. The size of the 
                       test set to create. Default to 0.1  ;
        - 'val_size': float. Strictly between 0.0 and 1.0. The size of the 
                      validation set to create. Default to 0.2  ;
        - 'random_state': optionnal, int. A random seed to initialize a 
                          pseudo-random sequence of numbers. Default to None0 ;
        - 'random_generator': optionnal, pseudo-random number generator 
                              object. Use to continue a pseudo-random sequence 
                              already initialized. Default to None.
    Return: the path to the balanced dataset directory.
    """

    balanced = num_images_per_cat > 0

    # Check the presence of the base directory where original images are stored
    source_dir = f'{root_dir}/base'

    if not Path(source_dir).exists():
        raise FileNotFoundError(f'{source_dir} not found')

    # Create directories to store the sets
    target_dir = f'{root_dir}/balanced' if balanced else f'{root_dir}/unbalanced'
    train_dir = f'{target_dir}/train'
    test_dir = f'{target_dir}/test'
    val_dir = f'{target_dir}/val'
    
    # Check whether the targeted directories exist or not, and create it if 
    # needed
    if not Path(target_dir).exists(): os.mkdir(Path(target_dir))
    if not Path(train_dir).exists(): os.mkdir(Path(train_dir))
    if not Path(test_dir).exists() : os.mkdir(Path(test_dir))
    if not Path(val_dir).exists() : os.mkdir(Path(val_dir))

    # Create a directory to store augmented images if needed
    if balanced:
        aug_dir = f'{root_dir}/augmented'
        if not Path(aug_dir).exists(): os.mkdir(Path(aug_dir))


    # Initialize a pseudo-random sequence
    if not random_generator:
        random_generator = random.Random(random_state)

    for current_dir, _, files_list in os.walk(source_dir):
        num_images = len(files_list)

        if num_images > 0:
            cat = split_path(current_dir)[-1]
            print(f'{cat}:', end='\t')
            print(f'init pop = {num_images}', end='\t')

            num_subset = num_images_per_cat if balanced else num_images

            if balanced and num_images < num_subset:
                # Create the target directory
                aug_dir = Path(f'{aug_dir}/{cat}')
                if not aug_dir.exists(): os.mkdir(aug_dir)
                image_data_augmentation(current_dir, aug_dir, num_subset, random_generator=random_generator)

                select_random_sample(
                    source_dir = aug_dir, 
                    target_dir = target_dir, 
                    cat = cat,
                    num_images_per_cat = num_subset, 
                    test_size = test_size, 
                    val_size = val_size, 
                    random_generator = random_generator
                )
            else:
                select_random_sample(
                    source_dir = current_dir, 
                    target_dir = target_dir, 
                    cat = cat,
                    num_images_per_cat = num_subset, 
                    test_size = test_size, 
                    val_size = val_size, 
                    random_generator = random_generator
                )
            
            print('done')

    def rrmdir(d):
        """
        Recursively remove folders in a directory, and finally the directory 
        itself.
        Parameter:
            - 'd': string. The directory path.
        Return: None
        """

        d = Path(d)
        for item in d.iterdir():
            if item.is_dir():
                rrmdir(item)
            else:
                item.unlink()
        d.rmdir()

    # Delete empty directories
    if test_size == 0.0: rrmdir(test_dir)
    if val_size == 0.0: rrmdir(val_dir)

    return target_dir

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main

if __name__ == "__main__":
    pass
