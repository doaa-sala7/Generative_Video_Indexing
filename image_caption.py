                                      # -*- coding: utf-8 -*-
"""create_caption.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zVckeHrVg6IsUyiXgRvGSILJgAW1ja7W
"""

import sys


# Base path for the image captioning related modules
base_path = '/content/drive/MyDrive/video_seg/image_captioning/'

# Insert the base path into the system path for module resolution
sys.path.insert(0, base_path)

# If your subdirectories also contain modules you wish to import, add them as well
sys.path.insert(0, base_path + 'utils')
sys.path.insert(0, base_path + 'utils/vocabulary.py')
sys.path.insert(0, base_path + 'utils/coco')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/cider/cider_scorer.py')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/tokenizer')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/cider')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/meteor')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/rouge')
sys.path.insert(0, base_path + 'utils/coco/pycocoevalcap/bleu')
sys.path.insert(0, base_path + 'utils/ilsvrc_2012_mean.npy')
sys.path.insert(0, base_path + 'utils/misc.py')
sys.path.insert(0, base_path + 'utils/nn.py')
sys.path.insert(0, base_path + '/image_captioning/utils/ilsvrc_2012_mean.npy')


import numpy as np
import tensorflow as tf
# from base_model import BaseModel
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data


def creat_caption():
 
    base_path = '/content/drive/MyDrive/video_seg/image_captioning/'

    tf.compat.v1.disable_eager_execution()

    config = Config()
    config.phase = 'test'
    config.beam_size = 3
    config.test_image_dir = "/content/drive/MyDrive/video_seg/framesxxxxx"
    config.train_caption_file = '/content/drive/MyDrive/video_seg/image_captioning/train/custom_coco_captions.json'

    npy_filename = 'models/291499.npy'
    path = base_path + npy_filename


    sess = tf.compat.v1.InteractiveSession()
    model = CaptionGenerator(config)


    model.load(sess, path)


    data, vocabulary = prepare_test_data(config)

    tf.compat.v1.get_default_graph().finalize()

    model.test(sess, data, vocabulary)