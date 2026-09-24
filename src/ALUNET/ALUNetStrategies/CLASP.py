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
from ALUNET.ALUNetStrategies.ALStrategy import ALStrategy
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.facilityLocationConditionalMutualInformation import FacilityLocationConditionalMutualInformationFunction
from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
from batchgenerators.utilities.file_and_folder_operations import load_json
from kneed import KneeLocator
from ct.ct import CTImage
soft1 = torch.nn.Softmax(dim=1)
from scipy.stats import entropy
from random import randint


class CLASP(ALStrategy):
    def __init__(self, name='CLASP'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        
        # sys.exit('TEST01')
        # data_query = alunet.man.datasets['query'].data
        # NumSamplesMaxMCD = 2000
        # previous=True
        # man=alunet.man 
        # folderDict = alunet.man.folderDict
        # NumSamples=50

        # self=strategy
        batch_size = 4
        alpha = 0.66
        vmax = 5
        beta0 = 1
        beta1 = 100
        eps = 1e-20
        #p_fg = 0.66

        # Select subset
        data_sub = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))

        # Predict uncertainty
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data_sub, batch_size=batch_size, single=True)
        NumBatches = math.ceil(len(data_sub)/dataloader_train.data_loader.batch_size)
        uc_c=[]
        uc=[]
        IDX=[]
        for b in tqdm(range(NumBatches), desc='Estimate uncertainty'):
            batch = next(dataloader_train)
            IDX = batch['idx'][:,0]
            datab = batch['data']
            datab = datab.to('cuda', non_blocking=True)
            out = net.model['unet'].network(datab)
            #pred = (soft1(out[0])>0.5)*1.0
            pred_p = soft1(out[0]).detach().cpu()
            pred_p = pred_p.clip(min=eps, max=1-eps)
            del out
            # !!! TODO
            if opts.dim==2:
                pred_uc = (-pred_p*torch.log2(pred_p)).sum(axis=1, keepdim=True)
                Nc = pred_p.shape[1]
                ent_c = pred_p * pred_uc.repeat(1, Nc, 1, 1)
                print('ent_c123', ent_c.shape)
                ent_c = ent_c.sum(axis=(2,3))
                ent = pred_uc.sum(axis=(1,2,3))
            else:
                pred_uc = (-pred_p*torch.log2(pred_p)).sum(axis=1, keepdim=True)
                Nc = pred_p.shape[1]
                print('pred_p123', pred_p.shape)
                print('pred_uc123', pred_uc.shape)
                ent_c = pred_p * pred_uc.repeat(1, Nc, 1, 1, 1)
                print('ent_c123', ent_c.shape)
                ent_c = ent_c.sum(axis=(2,3,4))
                ent = pred_uc.sum(axis=(1,2,3,4))

            for i,ID in enumerate(list(IDX)):
                data_sub[ID].F['uc_c']=ent_c[i]
                data_sub[ID].F['uc']=ent[i]

        # Check samples are not selected twice and stata and all and exclude background class in strata

        # Resort based on IDX
        uc =[]
        uc_c = []
        for s in data_sub:
            uc_c.append(s.F['uc_c'])
            uc.append(s.F['uc'])
        uc_c = torch.vstack(uc_c)
        uc = torch.vstack(uc)

        # Add gumble noise
        rng = np.random.default_rng()
        v = folderDict['version'] - 1 
        beta = np.exp((1 - (v/vmax)) * np.log(beta0) + (v/vmax) * np.log(beta1))
        uc_c =  np.log(uc_c.clamp_min(eps)) + rng.gumbel(0, beta**-1, size=uc_c.shape)
        uc =  np.log(uc.clamp_min(eps)) + rng.gumbel(0, beta**-1, size=uc.shape)

        # Sample based on startified uncertainty
        NumSamplesStrat = int(NumSamples*alpha)
        NumSamplesAll = NumSamples-NumSamplesStrat
        NumSamplesC = np.zeros(Nc, dtype=np.uint)
        for i in range(NumSamplesStrat):
            #pos = randint(0,Nc-1)
            pos = randint(1,Nc-1)
            NumSamplesC[pos] = NumSamplesC[pos]+1

        print('NumSamplesC123', NumSamplesC)

        idx_samples_strat = []
        #for c in range(Nc):
        for c in range(Nc):
            idx = torch.argsort(uc_c[:,c]).flip(dims=[0])
            k=0
            for i in list(idx):
                if k==NumSamplesC[c]:
                    break
                #if i not in idx_samples_strat:
                #    idx_samples_strat.append(int(i))
                #    k=k+1
                ii = int(i)
                if ii not in idx_samples_strat:
                    idx_samples_strat.append(ii)

        # Sample based on overall uncertainty
        idx_samples_all = []
        idx = torch.argsort(uc[:,0]).flip(dims=[0])
        k=0
        for i in list(idx):
            if k==NumSamplesAll:
                break
            
            #if i not in idx_samples_all:
            #if i not in idx_samples_strat:
            #    idx_samples_all.append(int(i))
            #    k=k+1
            ii = int(i)
            if ii not in idx_samples_strat:
                idx_samples_all.append(ii)
                
        idx_samples = idx_samples_strat + idx_samples_all

        print('idx_samples123', idx_samples)
        print('idx_samples_strat123', idx_samples_strat)
        print('idx_samples_all123', idx_samples_all)
        
        samples=[]
        for i in idx_samples[0:NumSamples]:
            samples.append(data_sub[i])

        del dataloader_train.data_loader.data_org
        del dataloader_train.data_loader.seg_org,
        del dataloader_train.data_loader.seg_prev_org
        del dataloader_train.data_loader.properties
        return samples


        #     idx_c = idx[0:NumSamplesC[c]]


        #     fg = []
        #     center = np.round(np.array(pred.shape)[2:]/2)
        #     if opts.dim==2:
        #         for i in range(pred.shape[0]):
        #             fg.append(bool(pred[i,:, int(center[0]), int(center[1])][1:].sum()>0))
        #     else:
        #         for i in range(pred.shape[0]):
        #             fg.append(bool(pred[i,:, int(center[0]), int(center[1]), int(center[2])][1:].sum()>0))

        #     # Set foreground in data
        #     for i,ID in enumerate(list(IDX)):
        #         data_sub[ID].F['uc']=fg[i]

        # del dataloader_train.data_loader.data_org
        # del dataloader_train.data_loader.seg_org,
        # del dataloader_train.data_loader.seg_prev_org
        # del dataloader_train.data_loader.properties
        
        # # Sample from unlabeled pool
        # fg_all = np.array([s.F['uc'] for s in data_sub])
        # idx_fg = list(np.where(fg_all==True)[0])
        # idx_bg = list(np.where(fg_all==False)[0])
        # NumSamplesFG = int(p_fg * NumSamples)
        # NumSamplesBG = NumSamples - NumSamplesFG
        # idx_fg_sel = random.sample(idx_fg, k=min(len(idx_fg), NumSamplesFG))
        # idx_bg_sel = random.sample(idx_bg, k=min(len(idx_bg), NumSamplesBG))
        # idx = list(idx_fg_sel + idx_bg_sel)
        # samples=[]
        # for i in idx[0:NumSamples]:
        #     samples.append(data_sub[i])
        # return samples
