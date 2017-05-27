import scipy
import scipy.misc
import os
from PIL import Image
import numpy as np
from Degradation import degrade
import random

class Dataset:

    def __init__(self, path, input_shape=(1024,1024), gt_shape=(1024,1024)):

        self.input_shape = input_shape
        self.gt_shape = gt_shape
        self.path = path
        self.cur_idx = 0
        self.max_idx = 0
        self.files = []
        if not os.path.exists(path):
            print("path is wrong!")
        else:
            for filename in os.listdir(path):
                if filename.endswith('.jpeg') or filename.endswith('.jpg'):
                    self.files.append(filename)
                    self.max_idx += 1

    def read_image(self, path, size):   # this function reads image as float64
        image = Image.open(path)
        ret = np.asarray(image.resize(size), dtype=np.uint8)
        return ret

    def change_format(self,image):
       return ((image*255)/np.max(image)).astype('uint8')

    def get_batch_inputs(self, path, idx):
        i_image = self.read_image(path, self.input_shape)
        # noise model
        if idx % 3 == 0:
            i_image = degrade(i_image, ['blur', 'noise', 'saturate', 'compress'])
        elif idx % 3 == 1:
            i_image = degrade(i_image, ['noise', 'saturate', 'compress'])
        else:
            i_image = degrade(i_image, ['downscale', 'noise', 'compress'])

        g_image = self.read_image(path, self.gt_shape)
        return i_image.astype(np.float32), g_image.astype(np.float32)

    def next_batch(self, batch_size):
        in_image=[]
        gt_image=[]
        cur_idx=self.cur_idx
        for i in range(batch_size):
            path = self.path+self.files[cur_idx]
            i_image, g_image = self.get_batch_inputs(path, cur_idx)
            in_image.append(i_image)
            gt_image.append(g_image)
            cur_idx = (cur_idx+1)%self.max_idx
        in_image = np.array(in_image)
        gt_image=np.array(gt_image)
        self.cur_idx=cur_idx # update for next batching
        return in_image,gt_image

    def random_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx = random.randint(0, self.max_idx-1)
        for i in range(batch_size):
            path = self.path + self.files[cur_idx]
            i_image, g_image = self.get_batch_inputs(path,cur_idx)
            in_image.append(i_image)
            gt_image.append(g_image)
            cur_idx = (cur_idx+1)%self.max_idx
        in_image = np.array(in_image)
        gt_image = np.array(gt_image)
        self.cur_idx = cur_idx # update for next batching
        return in_image, gt_image



