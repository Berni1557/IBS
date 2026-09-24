#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from nngeometry.layercollection import LayerCollection
import pandas as pd
from glob import glob
#from modules.XAL.strategies.ALStrategy import ALStrategy
from ALUNET.ALUNetStrategies.ALStrategy import ALStrategy
#from modules.UNetAL.ALUNet import UNetPatch
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.facilityLocationConditionalMutualInformation import FacilityLocationConditionalMutualInformationFunction
from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
from batchgenerators.utilities.file_and_folder_operations import load_json
from kneed import KneeLocator
from ct.ct import CTImage
soft1 = torch.nn.Softmax(dim=1)
from scipy.stats import entropy


class RANDOM66(ALStrategy):
    def __init__(self, name='RANDOM66'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        
        # self=strategy
        batch_size = 4
        p_fg = 0.66

        # Select subset
        data_sub = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))

        # Init model
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data_sub, batch_size=batch_size, single=True)
        NumBatches = math.ceil(len(data_sub)/dataloader_train.data_loader.batch_size)
        uc=[]
        for b in tqdm(range(NumBatches), desc='Estimate foreground'):
            batch = next(dataloader_train)
            IDX = batch['idx'][:,0]
            datab = batch['data']
            datab = datab.to('cuda', non_blocking=True)
            out = net.model['unet'].network(datab)
            pred = (soft1(out[0])>0.5)*1.0

            fg = []
            center = np.round(np.array(pred.shape)[2:]/2)
            if opts.dim==2:
                for i in range(pred.shape[0]):
                    fg.append(bool(pred[i,:, int(center[0]), int(center[1])][1:].sum()>0))
            else:
                for i in range(pred.shape[0]):
                    #fg.append(bool(pred[i,:, int(center[0]), int(center[1]), int(center[2])][1:].sum()>0))
                    fg.append(bool(pred[i][1:].sum()>0))

            # Set foreground in data
            for i,ID in enumerate(list(IDX)):
                data_sub[ID].F['uc']=fg[i]
            
            del pred
            del out

        del dataloader_train.data_loader.data_org
        del dataloader_train.data_loader.seg_org,
        del dataloader_train.data_loader.seg_prev_org
        del dataloader_train.data_loader.properties
        
        # Sample from unlabeled pool
        fg_all = np.array([s.F['uc'] for s in data_sub])
        idx_fg = list(np.where(fg_all==True)[0])
        NumSamplesFG = int(p_fg * NumSamples)
        idx_fg_sel = random.sample(idx_fg, k=min(len(idx_fg), NumSamplesFG))

        idx = np.arange(0,len(fg_all)).tolist()
        idx_rest = [x for x in idx if x not in idx_fg_sel]
        NumSamplesRest = NumSamples - NumSamplesFG
        idx_rest_sel = random.sample(idx_rest, k=min(len(idx_rest), NumSamplesRest))
        idx = list(idx_fg_sel + idx_rest_sel)

        # idx_bg = list(np.where(fg_all==False)[0])
        # NumSamplesFG = int(p_fg * NumSamples)
        # NumSamplesBG = NumSamples - NumSamplesFG
        # print('NumSamplesFG123', NumSamplesFG)
        # print('NumSamplesBG123', NumSamplesBG)
        #print('lenidx_fg', len(idx_fg), idx_fg)
        #print('lenidx_rest', len(idx_rest), idx_rest)
        #print('lenidx_fg_sel', len(idx_fg_sel), idx_fg_sel)
        #print('lenidx_rest_sel', len(idx_rest_sel), idx_rest_sel)
        # print('lenidx_bg', len(idx_bg), idx_bg)
        # idx_fg_sel = random.sample(idx_fg, k=min(len(idx_fg), NumSamplesFG))
        # idx_bg_sel = random.sample(idx_bg, k=min(len(idx_bg), NumSamplesBG))
        # idx = list(idx_fg_sel + idx_bg_sel)
        #print('idx123', len(idx), idx)
        samples=[]
        for i in idx[0:NumSamples]:
            samples.append(data_sub[i])
        #print('samples123', len(samples))
        return samples
