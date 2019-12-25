import os
import sys
import torch
import torch.utils.data as data 
import numpy as np
from PIL import Image              
import glob
import random
import cv2
random.seed(1143)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 定义一个函数，可以读取训练集元素的列表
def populate_train_list(orig_images_path, hazy_images_path):   # 定义函数：填写训练列表，有两个形参
	train_list = [] # 创建空列表
	val_list = []
	
	image_list_haze = glob.glob(hazy_images_path + "*.bmp")# data文件夹

	tmp_dict = {}# 创建空字典，大括号，字典是键(key)值(value)对应的

	for image in image_list_haze:      #遍历整个列表
		image = image.split("/")[-1]   #图片的整个名字NYU_8_4_1, fv_1_2

		key = image.split("_")[0] + "_"+ image.split("_")[1] +".bmp"  ###NYU2_8_4_1,key=NYU2_8,寻找一大类
		if key in tmp_dict.keys():   # 遍历字典中的所有键key()
			tmp_dict[key].append(image)  # 在末尾添加元素image。append动态创建列表
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)


	train_keys = []
	val_keys = []

	len_keys = len(tmp_dict.keys())  # 典中key的长度（应该是leibie数）
	for i in range(len_keys):
		if i < len_keys*9/10:
			train_keys.append(list(tmp_dict.keys())[i])
		else:
			val_keys.append(list(tmp_dict.keys())[i])  # 90%训练集


	for key in list(tmp_dict.keys()):

		if key in train_keys:
			for hazy_image in tmp_dict[key]:

				train_list.append([orig_images_path + hazy_image, hazy_images_path + hazy_image])
                                # 1to1,orig_images_path + hazy_image;1 to 10,orig_images_path + key

		else:
			for hazy_image in tmp_dict[key]:

				val_list.append([orig_images_path + hazy_image, hazy_images_path + hazy_image])



	random.shuffle(train_list)
	random.shuffle(val_list)

	return train_list, val_list


class dehazing_loader(data.Dataset):

	def __init__(self, orig_images_path, hazy_images_path, mode='train'):

		self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path) 

		if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list))
		else:
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list))


	def __getitem__(self, index):

		data_orig_path, data_hazy_path = self.data_list[index]

		data_orig = Image.open(data_orig_path)
		data_hazy = Image.open(data_hazy_path)

		data_orig = data_orig.resize((200,91), Image.ANTIALIAS)
		data_hazy = data_hazy.resize((200,91), Image.ANTIALIAS)

		data_orig = (np.asarray(data_orig)/255.0)
		data_hazy = (np.asarray(data_hazy)/255.0) 

		data_orig = torch.from_numpy(data_orig).float()
		data_hazy = torch.from_numpy(data_hazy).float()

		return data_orig.permute(2,0,1), data_hazy.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)
