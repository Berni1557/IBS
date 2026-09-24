#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join
import random
import torch
from tqdm import tqdm
import math
import numpy as np
#from nngeometry.layercollection import LayerCollection
import pandas as pd
from glob import glob
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
            
class ENTROPY(ALStrategy):
    def __init__(self, name='ENTROPY'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        #def query(self, opts, folderDict, man, data_query, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        # action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=batchsize, pred_class='XMaskPred', previous=previous, save_uc=False, NumSamplesMaxMCD=NumSamplesMaxMCD)
        # action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=batchsize, pred_class='XMaskPred', previous=previous, save_uc=False, NumSamplesMaxMCD=NumSamplesMaxMCD)
        # def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        # self=strategy
        # man=self.man
        # NumSamples=100
        # data_query=man.datasets['query'].data
        
        # Define parameters
        # layer_used='decoder'
        #NumSamplesMaxMCD=100000
        # NumSamplesQ=NumSamples
        # NumParams=10000
        # NumFIMMax=200                    # 500
        # save_uc=False
        #batch_size=16
        batch_size=4

        # Compute uncertainty
        data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
        self.getUncertainty(opts, folderDict, man, data_query, batch_size=batch_size, save_uc=save_uc, device='cuda', previous=True)

        uncertainty = [s.F['uc'].mean() for s in data_query]
        idx = list(np.argsort(uncertainty)[::-1])

        samples=[]
        for i in idx[0:NumSamples]:
            samples.append(data_query[i])
          
        return samples
    

    def getUncertainty(self, opts, folderDict, man, data_query, batch_size=None, save_uc=False, device='cuda', previous=True):

        batch_size = 1

        # init model
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        #enable_dropout(net.model['unet'].network, dropout_rate)
        
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data_query, batch_size=batch_size, single=True)
        NumBatches = math.ceil(len(data_query)/dataloader_train.data_loader.batch_size)
        uc=[]
        for b in tqdm(range(NumBatches), desc='Estimate uncertainty'):
            batch = next(dataloader_train)
            IDX = batch['idx'][:,0]
            datab = batch['data']
            datab = datab.to(device, non_blocking=True)
            
            # Iterate over monte carlo rounds
            #MCDrounds=[]
            #for r in range(NumMCDRounds):
            #    out = net.model['unet'].network(datab)
            #    for i in range(len(out)): out[i] = out[i].detach_().cpu()
            #    outs = soft1(out[0])
            #    MCDrounds.append(torch.unsqueeze(outs, dim=0).clone())
            out = net.model['unet'].network(datab)
            for i in range(len(out)): out[i] = out[i].detach_().cpu()
            outs = soft1(out[0])
            #print('outs123', outs.shape)
            ucMap = entropy(outs, axis=1, base=2)#
            ucMap = torch.from_numpy(ucMap)
            #print('ucMap1213', ucMap.shape)

            #MCDrounds = torch.vstack(MCDrounds)
            #ucMap = torch.var(MCDrounds, dim=0)
            #dimMean = tuple([i for i in range(2,2+opts.dim)])
            #uc = torch.mean(ucMap, dim=dimMean).numpy()

            dimMean = tuple([i for i in range(1,1+opts.dim)])
            #print('dimMean123', dimMean)
            uc = torch.mean(ucMap, dim=dimMean).numpy()

            #print('uc123', uc)
            
            # Set uncertainties in data
            for i,ID in enumerate(list(IDX)):
                data_query[ID].F['uc']=uc[i]
                if save_uc: 
                    data_query[ID].F['ucMap']=ucMap[i]
            

        del dataloader_train.data_loader.data_org
        del dataloader_train.data_loader.seg_org,
        del dataloader_train.data_loader.seg_prev_org
        del dataloader_train.data_loader.properties
