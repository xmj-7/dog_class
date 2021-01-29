import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Config import DataConfig
class DogDataSet(Dataset):
    def __init__(self,root,data_list_file,phase='train',input_size = 224):
        '''

        :param root: 训练集所在位置
        :param data_list_file:
        :param phase:
        :param input_size:输入图片的尺寸
        '''
        self.phase = phase
