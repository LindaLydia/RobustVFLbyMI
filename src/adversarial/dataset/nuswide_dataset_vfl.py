import os

import numpy as np
import random
import pandas as pd
from .nus_wide_data_util import get_labeled_data

class NUSWIDEDatasetVFL():

    def __init__(self, data_dir, data_type, poison_number, target_number=10, target_label=0, backdoor_scale=1.0):
        self.data_dir = data_dir
        self.target_label = target_label
        self.selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']
        self.class_num = 5

        print(self.selected_labels)
        if data_type == 'train':
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 60000, 'Train')
        else:
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 40000, 'Test')
        print(type(X_image), type(X_text), type(Y))

        X_text, poison_list = data_poison(X_text, (600 if data_type=="train" else 400))
        
        self.poison_images = [np.array(X_image).astype('float32')[poison_list], 
                              np.array(X_text).astype('float32')[poison_list]]
        self.poison_labels = np.argmax(np.array(Y),axis=1).astype('float32')[poison_list]
        
        self.poison_list = poison_list

        if data_type == 'train':
            self.x = [np.delete(np.array(X_image).astype('float32'), self.poison_list, axis=0),
                      np.delete(np.array(X_text).astype('float32'), self.poison_list, axis=0)]
            self.y = np.delete(np.argmax(np.array(Y),axis=1).astype('float32'), poison_list, axis=0)
        else:
            self.x = [np.array(X_image).astype('float32'), np.array(X_text).astype('float32')]
            self.y = np.argmax(np.array(Y), axis=1).astype('float32')

        self.target_list = random.sample(list(np.where(self.y==target_label)[0]), target_number)
        print(self.target_list)

        # check dataset
        print('dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)
        print('target data', self.y[self.target_list].shape, np.mean(self.y[self.target_list]), target_label)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()

    def get_poison_data(self):
        return self.poison_images, self.poison_labels

    def get_target_data(self):
        return [self.x[0][self.target_list], self.x[1][self.target_list]], self.y[self.target_list]

    def get_poison_list(self):
        return self.poison_list

    def get_target_list(self):
        return self.target_list


def need_poison_down_check_nuswide_vfl(images, backdoor_scale=1.0):
    # need_poison_list = [True if images[indx,-1]>backdoor_scale-(1e-6) else False\
    #                     for indx in range(len(images))]
    # return np.array(need_poison_list)
    return None

def data_poison(images, poison_number):
    poison_list = random.sample(range(images.shape[0]), poison_number)
    # images[poison_list,0,15,31] = target_pixel_value[0][0]
    # images[poison_list,0,14,30] = target_pixel_value[0][1]
    # images[poison_list,0,13,31] = target_pixel_value[0][2]
    # images[poison_list,0,15,29] = target_pixel_value[0][3]
    # images[poison_list,1,15,31] = target_pixel_value[1][0]
    # images[poison_list,1,14,30] = target_pixel_value[1][1]
    # images[poison_list,1,13,31] = target_pixel_value[1][2]
    # images[poison_list,1,15,29] = target_pixel_value[1][3]
    # images[poison_list,2,15,31] = target_pixel_value[2][0]
    # images[poison_list,2,14,30] = target_pixel_value[2][1]
    # images[poison_list,2,13,31] = target_pixel_value[2][2]
    # images[poison_list,2,15,29] = target_pixel_value[2][3]
    for idx in poison_list:
        images[idx,:] += np.random.randint(low=-1,high=2,size=(1000,),dtype=np.int64) #random from [low, high)
        np.putmask(images[idx,:], images[idx,:]>1, 1)
        np.putmask(images[idx,:], images[idx,:]<0, 0)
    return images, poison_list


if __name__ == '__main__':
    data_dir = "../../dataset/NUS_WIDE"

    # sel = get_top_k_labels(data_dir=data_dir, top_k=10)
    # print("sel", sel)
    # ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

    #sel_lbls = get_top_k_labels(data_dir, 81)
    #print(sel_lbls)

    train_dataset = NUSWIDEDatasetVFL(data_dir, 'train', backdoor_scale=1.0)
    print(train_dataset.y)

    #print(train_dataset.poison_list)

    res = need_poison_down_check_nuswide_vfl(train_dataset.x[1], 1.0)
    print(res.sum())


