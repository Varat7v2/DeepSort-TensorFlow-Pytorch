# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os, sys
import scipy.io
import yaml
import math
import cv2
import glob
import pickle
import shutil
import pandas as pd
from itertools import chain 
from model import ft_net
from PIL import Image

from myFROZEN_GRAPH_PERSON import FROZEN_GRAPH_INFERENCE


class PERSON_REID():

    def __init__(self):
        self.stride = 2
        self.nclasses = 751
        self.batchsize = 32
        # self.data_dir = 'dataset/mydata'
        # torch.cuda.set_device(0)
        # use_gpu = torch.cuda.is_available()
        self.model_structure = ft_net(self.nclasses, stride = self.stride)
        self.model = self.load_network(self.model_structure)
        self.model.classifier.classifier = nn.Sequential()
        # Change to test mode
        self.model = self.model.eval()
        # if use_gpu:
        #     model = model.cuda()

        self.data_transforms = transforms.Compose([
                transforms.Resize((256,128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir,x), self.data_transforms) for x in ['gallery']}
        # self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batchsize,
        #                                              shuffle=False, num_workers=16) for x in ['gallery']}

        # num_classes, num_files = fliplr_save('gallery')
        # # print(num_classes, num_files)
        # num_files_updated = list()
        # for i, x in enumerate(num_files):
        #     if i == 0:
        #         x_updated = x
        #     else:
        #         x_updated += x
        #     num_files_updated.append(x_updated)

        # cdict = dict()
        # lbound = 0
        # for cname, ubound in zip(num_classes, num_files_updated):
        #     cdict[cname] = [k for k in range(lbound, ubound)]
        #     lbound = ubound

    def load_network(self, network):
        model_path = 'models/person_reid.pth'
        network.load_state_dict(torch.load(model_path))
        return network

    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def serialize_dict(self, filename, mydict):
        # Serialize dictionary in binary format
        with open(filename, 'wb') as handle:
            pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_dict(self, filename):
        # Deserialize dictionary
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    def cosine_similarity(self, dict_encoding, current_encoding):
        return (np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

    def cosine_distance(self, dict_encoding, current_encoding):
        return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

    def eucledian_distance(self, dict_encoding, current_encoding):
        return np.sqrt(np.sum((dict_encoding - current_encoding) ** 2))

    # sort the images
    def sort_img(self, qf, gf):
        gf = np.asarray([f for f in gf])
        query = qf.reshape(-1,1)   #reshaping to shape (-1,1)
        score = np.dot(gf, query)
        score = np.squeeze(score, axis=1)
        # print('Query size: ', query.size())
        # print('Gallery feature: ', gf.size())
        # score = torch.mm(gf, query)
        # score = score.squeeze(1).cpu()
        # score = score.numpy()
        sorted_score = np.sort(score)[::-1] # in ascending order
        # print(sorted_score)

        index = np.argsort(score)  #from small to large
        index = index[::-1] #reverse the sequences (large to small)
        
        return sorted_score, index

    ######################################################################
    # Extract feature
    # ----------------------
    # def extract_feature(self, dataloaders):
    #     for data in dataloaders:
    #         img, label = data
    #         n, c, h, w = img.size()
    #         input_img = Variable(img)
    #         # Model output --> feature vectors
    #         outputs = model(input_img)
    #         self.ff += outputs        
    #         # Model output --> normalized feature vectors
    #         fnorm = torch.norm(self.ff, p=2, dim=1, keepdim=True)
    #         self.ff = self.ff.div(fnorm.expand_as(self.ff))
    #         self.features = torch.cat((features, self.ff.data.cpu()), 0)
        
    #     return self.features

    # def extract_feature_imgTensor(self, persons):
    #     for person in persons:
    #         t = time.time()
    #         pil_img = Image.fromarray(cv2.cvtColor(person['cropped'], cv2.COLOR_BGR2RGB))
    #         img = self.data_transforms(pil_img).float()
    #         img = img.unsqueeze_(0)
    #         n, c, h, w = img.size()
    #         features = torch.FloatTensor()
    #         ff = torch.FloatTensor(n,512).zero_()
    #         ms = [math.sqrt(float(1))]
    #         input_img = Variable(img)

    #         # Model output --> feature vectors
    #         outputs = self.model(input_img)
    #         ff += outputs        
    #         # Model output --> normalized feature vectors
    #         fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    #         ff = ff.div(fnorm.expand_as(ff))
    #         features = torch.cat((features, ff.data.cpu()), 0)
    #         # print('Time taken for extracting embedding per person: {:2f} milliseconds'.format((time.time()-t)*1000))
    #     return features

    def extract_feature_imgTensor(self, persons):
        features = torch.FloatTensor()
        for person in persons:
            t = time.time()
            pil_img = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
            img = self.data_transforms(pil_img).float()
            img = img.unsqueeze_(0)
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n,512).zero_()
            ms = [math.sqrt(float(1))]
            input_img = Variable(img)

            # Model output --> feature vectors
            outputs = self.model(input_img)
            ff += outputs        
            # Model output --> normalized feature vectors
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff.data.cpu()), 0)
            # print('Time taken for extracting embedding per person: {:2f} milliseconds'.format((time.time()-t)*1000))
        return features.cpu().detach().numpy()

    # def fliplr_save(folder):
    #     global num_classes, num_files_inClass
    #     num_classes = list()
    #     num_files_inClass = list()
    #     for mydir in sorted(os.listdir(os.path.join(data_dir, folder))):
    #         num_classes.append(mydir)
    #         num_files = 0
    #         for img in glob.glob(os.path.join(data_dir, folder, mydir) + '/*'):
    #             # filename = img.split('/')[-1].split('.')[0]
    #             # if folder == 'query':
    #                 # flip_check = filename.split('/')[-1].split('_')[-1]
    #                 # if flip_check != 'flipped':
    #                 #     frame = cv2.imread(img)
    #                 #     frame = np.fliplr(frame)
    #                 #     cv2.imwrite(os.path.join(data_dir, folder, mydir)+'/'+filename+'_flipped.png', frame)
    #                 #     # num_files += 1
    #             num_files += 1
    #         num_files_inClass.append(num_files)
    #     return num_classes, num_files_inClass

def main():

    EXTRACT_GALLERY_FEATURES = True
    PATH_TO_CKPT_PERSON = 'models/frozen_graph_mobilenet_v2.pb'

    frozen_person = FROZEN_GRAPH_INFERENCE(PATH_TO_CKPT_PERSON)
    reid = PERSON_REID()

    print('-------   INFERENCE  INFO -----------')
    ## FROZEN MODEL RUN --> PERSON DETECTION
    gallery_folder = 'dataset/mydata/gallery'
    query_folder = 'dataset/mydata/query'
    cam = cv2.VideoCapture('videos/cctv.mp4')
    # cam = cv2.VideoCapture(0)
    im_width = int(cam.get(3))
    im_height = int(cam.get(4))
    
    gallery_df = pd.DataFrame()
    while True:
        t1 = time.time()
        ret, frame = cam.read()
        if ret == False:
            print('No frame detected!!!')
            break
        
        persons = frozen_person.run_frozen_graph(frame, im_width, im_height)
        persons_id = [k for k in range(len(persons))]

        if len(persons) > 0 :
            # EXTRACT FEATURE OF QUERY PERSON
            with torch.no_grad():
                # query_features = extract_feature(model, dataloaders_query['query'])
                query_features = reid.extract_feature_imgTensor(persons)
            print('Query Features size: ', query_features.size())

            query_lst = list()
            for idx, person, qf in zip(persons_id, persons, query_features):
                query_dict = {'person_id': idx,
                              'person_cropped': person['cropped'],
                              'person_embedding': qf.detach().cpu().numpy()}
                query_lst.append(query_dict)

            query_df = pd.DataFrame(query_lst)
            # print(query_df.head())
            # query_arr = np.vstack(query_df.loc[query_df['person_id']==0]['person_embedding']).astype(np.float)
            # print(query_arr.shape)
            # sys.exit(0)

            if gallery_df.shape[0] > 0:
                # print('Gallery IDs: ', gallery_df['person_id'].iloc[-1]+1)
                # COMPARISION BY SORTING ALGORITHM
                person_ids = list()
                person_id_scores = list()
                for person, query_embedding in zip(persons, query_df['person_embedding']):
                    # print(query_embedding.shape)
                    score_list, index_list = reid.sort_img(query_embedding, gallery_df['person_embedding'])

                    if score_list[0] > 0.7:
                        name = index_list[0]
                    else:
                        name = 'Unknown'
                        query_intruder = {'person_id': gallery_df['person_id'].iloc[-1]+1,
                                          'person_cropped': person['cropped'],
                                          'person_embedding': query_embedding}
                        gallery_df = gallery_df.append(pd.DataFrame([query_intruder]))
                        # gallery_df = pd.concat([gallery_df, query_intruder])
                        # query_embedding.unsqueeze_(0)
                        # gallery_features = torch.cat((gallery_features, query_embedding), dim=0)
                    
                    cv2.rectangle(frame, (person['left'], person['top']), 
                        (person['right'], person['bottom']), (0, 255, 0), 1, 8)
                    cv2.putText(frame, 'ID: {}/{:.2f}'.format(name, score_list[0]), 
                        (person['left']+5, person['top']-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            else:
                # gallery_features = query_features
                gallery_df = query_df

            print('Gallery size: ', gallery_df.shape)

        fps = 1 / (time.time() - t1)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()