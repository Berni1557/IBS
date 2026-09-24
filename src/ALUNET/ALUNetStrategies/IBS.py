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
from ct.ct import CTImage, CTRef
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.facilityLocationConditionalMutualInformation import FacilityLocationConditionalMutualInformationFunction
from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction
from batchgenerators.utilities.file_and_folder_operations import load_json
from kneed import KneeLocator
from ct.ct import CTImage
from sklearn.neighbors import LocalOutlierFactor
from torch.distributions import Categorical
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt
soft1 = torch.nn.Softmax(dim=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# def enable_dropout(model, drop_rate=0.5):
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             m.p=drop_rate
#             m.train()
#         if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
#             m.p=drop_rate
#             m.train()
            
# def disable_dropout(model):
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             m.p=0.0
#             m.eval()
#         if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
#             m.p=0.0
#             m.eval()

# def similarity_fim(data0, data1, FI):
#     S=torch.zeros((len(data0), len(data1)))
#     G0 = torch.vstack([s.F['grad']*FI.cpu() for s in data0])
#     G1 = torch.vstack([s.F['grad'] for s in data1]).transpose(0,1)
#     S = torch.matmul(G0,G1)/len(FI)
#     Sn = S.numpy()
#     return Sn
             
            
class IBS(ALStrategy):
    def __init__(self, name='IBS'):
        self.name = name
        self.dtype=torch.FloatTensor

    # def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False):
        
    #     # self=strategy
    #     # NumSamples=100
    #     # data_query=man.datasets['query'].data
        
    #     # Define parameters
    #     layer_used='decoder'
    #     NumSamplesMaxMCD=1000000
    #     NumSamplesQ=NumSamples
    #     NumParams=10000
    #     NumFIMMax=200                    # 500
    #     save_uc=False
    #     dropout_rate=0.1
    #     NumMCDRounds=10
    #     batch_size=16

    #     # Compute uncertainty
    #     data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
    #     self.getUncertainty(opts, folderDict, man, data_query, batch_size=batch_size, save_uc=save_uc, device='cuda', previous=True, NumMCDRounds=NumMCDRounds, dropout_rate=dropout_rate)
        
    #     # Extract weights
    #     classWeight = self.getClassWeighting(opts, man)

    #     # Weighted uncertainty
    #     label_ignore = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
    #     uc = np.array([s.F['uc'] for s in data_query])
    #     ucw = np.zeros(uc.shape[0])
    #     for i in range(label_ignore):
    #         ucw = ucw+classWeight[i]*uc[:,i]
    #     prop = ucw/ucw.sum()
        
    #     # Get Network and layers
    #     net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous=True)

    #     # Compute F
    #     data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False, p=prop))
    #     FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous=True)

    #     # Compute gradients of data_QU and data_uc
    #     self.grad_data(opts, folderDict, man, net, data_query, layer_collection, previous, idxG=idxG)

    #     # Estimate number of cluster
    #     x = np.array([i for i in range(len(uc))])
    #     y = sorted(ucw)[::-1]
    #     kneedle = KneeLocator(x, y, S=10.0, curve="convex", direction="decreasing")
    #     NumSamplesQ = int(min(max(kneedle.knee, NumSamplesQ), len(data_query)))

    #     # Submodular subset selection
    #     data_Q = list(np.random.choice(data_query, size=min(NumSamplesQ, len(data_query)), replace=False, p=prop))
    #     query_sijs = similarity_fim(data_query, data_Q, FI)
    #     n = len(data_query)
    #     num_queries = len(data_Q)
    #     optimizer = 'StochasticGreedy'
    #     stopIfZeroGain = False
    #     stopIfNegativeGain = False
    #     verbose = False
    #     show_progress = False
    #     epsilon = 0.1
    #     budget = NumSamples
    #     queryDiversityEta = 0.1
    #     obj = FacilityLocationVariantMutualInformationFunction(n, num_queries, query_sijs=query_sijs, data=None, queryData=None, metric=None, queryDiversityEta=queryDiversityEta)
    #     greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #     greedyIndices = [x[0] for x in greedyList]
    #     samples = [data_query[i] for i in greedyIndices]
        
    #     for s in samples:
    #         print(s.F['imagename'], s.F['lbs_org'][0])
        
    #     # Delete gradients and uncertainty
    #     for s in samples + data_query + data_Q + data_F:
    #         if 'grad' in s.F: del s.F['grad']
    #         if 'uc' in s.F: del s.F['uc']
  
    #     return samples
    
    # def select_manual(self, opts, folderDict, man, data_query, data_action, NumSamples, previous=True):
        
    #     # self=strategy
        
    #     # data_query=alunet.man.datasets['query'].data
    #     # data_action=data_action_select
    #     # man=alunet.man
    #     # folderDict = man.folderDict
    #     # previous=True
        
    #     # Define parameters
    #     layer_used='decoder'
    #     #NumSamplesMaxMCD=100000
    #     NumSamplesMaxMCD=3000
    #     NumSamplesQ=NumSamples
    #     NumParams=10000
    #     NumFIMMax=200                    # 500
    #     save_uc=False
    #     dropout_rate=0.1
    #     NumMCDRounds=10
    #     batch_size=16
    #     #method = 'FacilityLocationConditionalMutualInformationFunction'
    #     method = 'LogDeterminantConditionalMutualInformationFunction'

    #     data_pos = [s for s in data_action if (s.action.info['class']==1)]
    #     data_neg = [s for s in data_action if (s.action.info['class']==2)]
        
    #     # Get Network and layers
    #     net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous=True)
        


    #     if opts.fastmode:
    #         FI = data_query[0].F['FI']
    #         idxG = data_query[0].F['idxG']
    #         data = [s for s in data_query if ('uc' in s.F) and np.any(~np.isnan(s.F['uc']))]
            
            
    #         print('lendata123', len(data))
    #         print('grad123', data[0].F['grad'].shape)
    #         #self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, previous, idxG=idxG)
    #         #self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, previous, idxG=idxG)
            
    #         #dict_grad = {'entropy': False, 'entropy_map': False, 'previous': False, 'use_mask': False, 'batch_size': 4}
    #         dict_grad = {'entropy': False, 'entropy_map': False, 'previous': False, 'use_mask': False, 'batch_size': 4}
            
            
    #         self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #     else:
    #         data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False))
    #         FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, dict_grad=dict_grad)
    #         data = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
    #         self.grad_data(opts, folderDict, man, net, data, layer_collection, previous, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, previous, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, previous, idxG=idxG, dict_grad=dict_grad)


            
    #     # Compute gradients of data_QU and data_uc
    #     # if not np.all([(('grad' in s.F) and (isinstance(s.F['grad'], torch.DoubleTensor))) for s in data_query]):
    #     #     self.grad_data(opts, folderDict, man, net, data, layer_collection, previous, idxG=idxG)
    #     #     self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, previous, idxG=idxG)
    #     #     self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, previous, idxG=idxG)
    #     # else:
    #     #     #for s in data_query: s.F['grad']=torch.from_numpy(s.F['grad'])
    #     #     pass
        
    #     # Option to switch positive and negative
    #     if False:
    #         data_tmp = data_pos
    #         data_pos = data_neg
    #         data_neg = data_tmp
            
            
    #     # Submodular subset selection
    #     if method=='FacilityLocationConditionalMutualInformationFunction':
    #         data_sijs = similarity_fim(data, data, FI)
    #         query_sijs = similarity_fim(data, data_pos, FI)
    #         private_sijs  = similarity_fim(data, data_neg, FI)
    #         n = len(data)
    #         num_queries = len(data_pos)
    #         num_privates = len(data_neg)
    #         optimizer = 'StochasticGreedy'
    #         stopIfZeroGain = False
    #         stopIfNegativeGain = False
    #         verbose = False
    #         show_progress = False
    #         epsilon = 0.1
    #         budget = NumSamples
    #         magnificationEta  = 1.0
    #         privacyHardness = 1.0
    #         lambdaVal = 0.01
    #         obj = FacilityLocationConditionalMutualInformationFunction(n, num_queries, num_privates, data_sijs=data_sijs, query_sijs=query_sijs, private_sijs=private_sijs, data=None, queryData=None, privateData=None, metric=None, magnificationEta =magnificationEta , privacyHardness=privacyHardness)
    #         greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #         greedyIndices = [x[0] for x in greedyList]
    #         samples = [data[i] for i in greedyIndices]
            
    #     elif method=='LogDeterminantConditionalMutualInformationFunction':
    #         data_sijs = similarity_fim(data, data, FI)
    #         query_sijs = similarity_fim(data, data_pos, FI)
    #         private_sijs  = similarity_fim(data, data_neg, FI)
    #         query_query_sijs = similarity_fim(data_pos, data_pos, FI)
    #         private_private_sijs = similarity_fim(data_neg, data_neg, FI)
    #         query_private_sijs = similarity_fim(data_pos, data_neg, FI)
    #         n = len(data)
    #         num_queries = len(data_pos)
    #         num_privates = len(data_neg)
    #         optimizer = 'StochasticGreedy'
    #         stopIfZeroGain = False
    #         stopIfNegativeGain = False
    #         verbose = True
    #         show_progress = True
    #         epsilon = 0.01
    #         budget = NumSamples
    #         magnificationEta  = 1.0
    #         privacyHardness = 1.0
    #         lambdaVal = 0.01
    #         obj = LogDeterminantConditionalMutualInformationFunction(n, num_queries, num_privates, data_sijs=data_sijs, query_sijs=query_sijs, query_query_sijs=query_query_sijs, private_sijs=private_sijs, private_private_sijs=private_private_sijs, query_private_sijs=query_private_sijs, data=None, queryData=None, privateData=None, metric=None, magnificationEta =magnificationEta , privacyHardness=privacyHardness, lambdaVal=lambdaVal)
    #         greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #         greedyIndices = [x[0] for x in greedyList]
    #         samples = [data[i] for i in greedyIndices]
            
    #     else:
    #         pass
        
    #     # # Delete gradients
    #     # for s in samples + data_pos + data_neg + data_F + data_query:
    #     #     if 'grad' in s.F: del s.F['grad'] 
    #     #     if 'FI' in s.F: del s.F['FI'] 

    #     return samples

    # def select_manual2(self, opts, folderDict, man, data_query, data_action, NumSamples, previous=True):
        
    #     # self=strategy
        
    #     # data_query=alunet.man.datasets['query'].data
    #     # data_action=data_action_select
    #     # man=alunet.man
    #     # folderDict = man.folderDict
    #     # previous=True
        
    #     # Define parameters
    #     layer_used='decoder'
    #     #NumSamplesMaxMCD=100000
    #     NumSamplesMaxMCD=5000
    #     NumSamplesQ=NumSamples
    #     NumParams=10000
    #     NumFIMMax=200                    # 500
    #     save_uc=False
    #     dropout_rate=0.1
    #     NumMCDRounds=10
    #     batch_size=16
    #     #method = 'FacilityLocationConditionalMutualInformationFunction'
    #     #method = 'LogDeterminantConditionalMutualInformationFunction'
    #     #method = 'FacLoc'
    #     method = 'LogDet'
        
    #     # Compute uncertainty
    #     # data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
    #     # data_pos = [s for s in data_action if (s.action.info['classified'] and  s.action.info['class']==1)]
    #     # data_neg = [s for s in data_action if (s.action.info['classified'] and  s.action.info['class']==2)]
    #     #data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
        
    #     data_pos = [s for s in data_action if (s.action.info['class']==1)]
    #     data_neg = [s for s in data_action if (s.action.info['class']==2)]
        
    #     # Get Network and layers
    #     net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous=True)
        

    #     # !!! Check for alternatives
    #     # if len(idxG)==0:
    #     #     params = layer_collection.get_parameters_BF(net.model['unet'].network)
    #     #     num_params = int(np.sum([x.numel() for x in params]))
    #     #     population = [x for x in range(num_params)]
    #     #     idxG = np.random.choice(population, size=NumParams, replace=False)
    #     #     FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous=True, idxG=idxG)

    #     # Compute F
    #     # if opts.fastmode:
    #     #     FI=data_query[0].F['FI']
    #     #     idxG=data_query[0].F['idxG']
    #     # else:
    #     #     data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False))
    #     #     FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous=True)
        
    #     if opts.fastmode:
    #         FI = data_query[0].F['FI']
    #         idxG = data_query[0].F['idxG']
    #         data = [s for s in data_query if ('uc' in s.F) and np.any(~np.isnan(s.F['uc']))]
            
            
    #         print('lendata123', len(data))
    #         #print('uc123', data[0].F['uc'])
    #         #print('grad123', data[0].F['grad'])
    #         #print('grad1234', data[0].F['grad'].shape)
    #         #self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, previous, idxG=idxG)
    #         #self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, previous, idxG=idxG)
    #         dict_grad = {'entropy': False, 'entropy_map': False, 'previous': True, 'use_mask': False, 'batch_size': 4}
    #         self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #         # !!!
    #         # CHECK why necessary!
    #         data = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
    #         self.grad_data(opts, folderDict, man, net, data, layer_collection, idxG=idxG, dict_grad=dict_grad)
            
    #         # dict_grad = {'entropy': False, 'entropy_map': False, 'previous': True, 'use_mask': False, 'batch_size': 4}
    #         # self.grad_data(opts, folderDict, man, net, sl, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #     else:
    #         dict_grad = {'entropy': False, 'entropy_map': False, 'previous': True, 'use_mask': False, 'batch_size': 4}
    #         data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False))
    #         #FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous=True)
    #         FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, dict_grad=dict_grad)
    #         data = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
    #         self.grad_data(opts, folderDict, man, net, data, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, idxG=idxG, dict_grad=dict_grad)
    #         self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, idxG=idxG, dict_grad=dict_grad)

    #         # FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_pos, layer_used, NumParams, dict_grad=dict_grad)
            
    #     # Compute gradients of data_QU and data_uc
    #     # if not np.all([(('grad' in s.F) and (isinstance(s.F['grad'], torch.DoubleTensor))) for s in data_query]):
    #     #     self.grad_data(opts, folderDict, man, net, data, layer_collection, previous, idxG=idxG)
    #     #     self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, previous, idxG=idxG)
    #     #     self.grad_data(opts, folderDict, man, net, data_neg, layer_collection, previous, idxG=idxG)
    #     # else:
    #     #     #for s in data_query: s.F['grad']=torch.from_numpy(s.F['grad'])
    #     #     pass
        
    #     # Option to switch positive and negative
    #     if False:
    #         data_tmp = data_pos
    #         data_pos = data_neg
    #         data_neg = data_tmp
            
    #     if len(data_neg)>0:
    #         # Submodular subset selection
    #         if method=='FacLoc':
    #             data_sijs = similarity_fim(data, data, FI)
    #             query_sijs = similarity_fim(data, data_pos, FI)
    #             private_sijs  = similarity_fim(data, data_neg, FI)
    #             n = len(data)
    #             num_queries = len(data_pos)
    #             num_privates = len(data_neg)
    #             optimizer = 'StochasticGreedy'
    #             stopIfZeroGain = False
    #             stopIfNegativeGain = False
    #             verbose = False
    #             show_progress = False
    #             epsilon = 0.1
    #             budget = NumSamples
    #             magnificationEta  = 1.0
    #             privacyHardness = 1.0
    #             lambdaVal = 0.01
    #             obj = FacilityLocationConditionalMutualInformationFunction(n, num_queries, num_privates, data_sijs=data_sijs, query_sijs=query_sijs, private_sijs=private_sijs, data=None, queryData=None, privateData=None, metric=None, magnificationEta =magnificationEta , privacyHardness=privacyHardness)
    #             greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #             greedyIndices = [x[0] for x in greedyList]
    #             samples = [data[i] for i in greedyIndices]
                
    #         elif method=='LogDet':
    #             data_sijs = similarity_fim(data, data, FI)
    #             query_sijs = similarity_fim(data, data_pos, FI)
    #             private_sijs  = similarity_fim(data, data_neg, FI)
    #             query_query_sijs = similarity_fim(data_pos, data_pos, FI)
    #             private_private_sijs = similarity_fim(data_neg, data_neg, FI)
    #             query_private_sijs = similarity_fim(data_pos, data_neg, FI)
    #             n = len(data)
    #             num_queries = len(data_pos)
    #             num_privates = len(data_neg)
    #             optimizer = 'StochasticGreedy'
    #             stopIfZeroGain = False
    #             stopIfNegativeGain = False
    #             verbose = True
    #             show_progress = True
    #             epsilon = 0.01
    #             budget = NumSamples
    #             magnificationEta  = 1.0
    #             privacyHardness = 1.0
    #             lambdaVal = 0.01
    #             obj = LogDeterminantConditionalMutualInformationFunction(n, num_queries, num_privates, data_sijs=data_sijs, query_sijs=query_sijs, query_query_sijs=query_query_sijs, private_sijs=private_sijs, private_private_sijs=private_private_sijs, query_private_sijs=query_private_sijs, data=None, queryData=None, privateData=None, metric=None, magnificationEta =magnificationEta , privacyHardness=privacyHardness, lambdaVal=lambdaVal)
    #             greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #             greedyIndices = [x[0] for x in greedyList]
    #             samples = [data[i] for i in greedyIndices]
    #         else:
    #             pass
    #     else:
    #         # Submodular subset selection
    #         if method=='FacLoc':
    #             data_sijs = similarity_fim(data, data, FI)
    #             query_sijs = similarity_fim(data, data_pos, FI)
    #             #private_sijs  = similarity_fim(data, data_neg, FI)
    #             n = len(data)
    #             num_queries = len(data_pos)
    #             #num_privates = len(data_neg)
    #             optimizer = 'StochasticGreedy'
    #             stopIfZeroGain = False
    #             stopIfNegativeGain = False
    #             verbose = False
    #             show_progress = False
    #             epsilon = 0.1
    #             budget = NumSamples
    #             magnificationEta  = 1.0
    #             privacyHardness = 1.0
    #             lambdaVal = 0.01
    #             obj = FacilityLocationMutualInformationFunction(n, num_queries, data_sijs=data_sijs, query_sijs=query_sijs, data=None, queryData=None, metric=None, magnificationEta=magnificationEta)
    #             greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #             greedyIndices = [x[0] for x in greedyList]
    #             samples = [data[i] for i in greedyIndices]
                
    #         elif method=='LogDet':
    #             data_sijs = similarity_fim(data, data, FI)
    #             query_sijs = similarity_fim(data, data_pos, FI)
    #             #private_sijs  = similarity_fim(data, data_neg, FI)
    #             query_query_sijs = similarity_fim(data_pos, data_pos, FI)
    #             #private_private_sijs = similarity_fim(data_neg, data_neg, FI)
    #             #query_private_sijs = similarity_fim(data_pos, data_neg, FI)
    #             n = len(data)
    #             num_queries = len(data_pos)
    #             #num_privates = len(data_neg)
    #             optimizer = 'StochasticGreedy'
    #             stopIfZeroGain = False
    #             stopIfNegativeGain = False
    #             verbose = False
    #             show_progress = False
    #             epsilon = 0.002
    #             budget = NumSamples
    #             magnificationEta  = 1.0
    #             privacyHardness = 1.0
    #             lambdaVal = 0.01
    #             #obj = LogDeterminantConditionalMutualInformationFunction(n, num_queries, num_privates, data_sijs=data_sijs, query_sijs=query_sijs, query_query_sijs=query_query_sijs, private_sijs=private_sijs, private_private_sijs=private_private_sijs, query_private_sijs=query_private_sijs, data=None, queryData=None, privateData=None, metric=None, magnificationEta =magnificationEta , privacyHardness=privacyHardness, lambdaVal=lambdaVal)
    #             obj = LogDeterminantMutualInformationFunction(n, num_queries, data_sijs=data_sijs, query_sijs=query_sijs, query_query_sijs=query_query_sijs, data=None, queryData=None, metric=None, magnificationEta =magnificationEta , lambdaVal=lambdaVal)
    #             greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
    #             greedyIndices = [x[0] for x in greedyList]
    #             samples = [data[i] for i in greedyIndices]
    #         else:
    #             pass
        
    #     # Delete gradients
    #     for s in samples + data_pos + data_neg + data_F + data_query:
    #         if 'grad' in s.F: del s.F['grad'] 
    #         if 'FI' in s.F: del s.F['FI'] 
        
    #     # for i,s0 in enumerate(data_pos):
    #     #     for j,s1 in enumerate(data):
    #     #         if s0.F['ID']==s1.F['ID']:
    #     #             sys.exit()
        
    #     return samples
    

    # def check_labeled(self, opts, folderDict, man, data_query, data_train, NumSamplesCheck, previous=True):
        
    #     # self=strategy
        
    #     # data_query=alunet.man.datasets['query'].data
    #     # man=alunet.man
    #     # folderDict=alunet.man.folderDict
        
    #     # Replace labels
    #     #ref = CTRef('/mnt/HHD/data/UNetAL/nnunet/nnUNet_raw/Dataset204_CTA/labelsTr/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559.nii.gz')
    #     #ref = CTRef('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset204_CTA/gt_segmentations/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559.nii.gz')
        
    #     #arr = ref.ref()
    #     #arr = np.load('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset204_CTA/nnUNetPlans_2d/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559_seg.npy')
    #     #arr[333, 250:300, 270:320]=0
    #     #arr[333, 270:320, 250:300]=0
    #     #arr[0,333, 270:350, 250:350]=0
    #     #np.save('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset204_CTA/nnUNetPlans_2d/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559_seg.npy', arr)
    #     #ref.setRef(arr)
    #     #ref.save('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset204_CTA/gt_segmentations/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559.nii.gz')
        
    #     #ref.save('/mnt/HHD/data/UNetAL/nnunet/nnUNet_raw/Dataset204_CTA/labelsTr/23-LIV-0013_1.2.840.113704.7.1.0.15041132133144208.1475763008.559.nii.gz')
        
        
    #     # Define parameters
    #     layer_used='decoder'
    #     NumSamplesMaxMCD=100
    #     #NumSamplesTest=1000
    #     NumSamplesTest=100
    #     NumParams=10000
    #     #NumFIMMax=200                 # 500
    #     NumFIMMax=10   
    #     save_uc=False
    #     dropout_rate=0.1
    #     NumMCDRounds=10
    #     batch_size=16
        
    #     data_query = man.datasets['query'].data
    #     data_query = random.sample(data_query, k=min(len(data_query), NumSamplesTest))
    #     data_train = man.datasets['train'].data

    #     # Get Network and layers
    #     net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous=previous)
        
    #     # Compute F
    #     #data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False))
    #     data_F = list(np.random.choice(data_train, size=min(NumFIMMax, len(data_train)), replace=False))
    #     dict_grad = {'entropy': False, 'entropy_map': False, 'previous': previous, 'use_mask': False, 'batch_size': 4}
    #     FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, dict_grad=dict_grad)
        
    #     from torch.nn.functional import normalize
    #     dict_grad = {'entropy': False, 'entropy_map': False, 'previous': previous, 'use_mask': False, 'batch_size': 4}
    #     self.grad_data(opts, folderDict, man, net, data_train, layer_collection, previous=True, idxG=idxG, use_mask=True)
    #     grad0 = torch.vstack([s.F['grad'] for s in data_train])
    #     self.grad_data(opts, folderDict, man, net, data_train, layer_collection, previous=True, idxG=idxG, use_mask=False)
    #     grad1 = torch.vstack([s.F['grad'] for s in data_train])
        
    #     grad0n= normalize(grad0, p=2.0, dim = 1)
    #     grad1n= normalize(grad1, p=2.0, dim = 1)
        
    #     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #     Xs = cos(grad0n, grad1n)
        
    #     idx = np.argsort(Xs.numpy())
    #     data_check = [data_train[i] for i in idx[0:NumSamplesCheck]]
        
    #     # Compute gradients of data_QU and data_uc
    #     #self.grad_data(opts, folderDict, man, net, data_train[0:20], layer_collection, previous=True, idxG=idxG, use_mask=True)
    #     #self.grad_data(opts, folderDict, man, net, data_query, layer_collection, previous=True, idxG=idxG, use_mask=False)
        
    #     #im=CTImage('/mnt/hpc_XAL/data/UNetAL/nnunet/nnUNet_raw/Dataset704_CTA17C/labelsTr/CADMAN-305_1.2.392.200036.9116.2.2426555318.1415687287.31.1289700022.1.nii.gz')
        
    #     # ###########################
    #     # # Compute similarity matrix
    #     # X = similarity_fim(data_train, data_query, FI)
    #     # #X = similarity_fim(data_train, data_train, FI)
    #     # #np.fill_diagonal(X, 0.0)
    #     # Xabs = np.abs(X)
        
    #     # #l0 = [torch.linalg.vector_norm(s.F['grad']) for s in data_train]
    #     # #l1 = [torch.linalg.vector_norm(s.F['grad']) for s in data_train]
        
        

    #     # Xs =  Xabs.sum(axis=1)
    #     # idx = np.argsort(Xs)[::-1]
    #     # data_check = [data_train[i] for i in idx[0:NumSamplesCheck]]
        
        
    #     # ################################

    #     return data_check
        

    # def getClassWeighting(self, opts, man):
    #     data_train = man.datasets['train'].data
    #     imagenames = np.unique([s.F['imagename'] for s in data_train])
    #     label_ignore = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
    #     weight = np.zeros(label_ignore)
    #     for im in imagenames:
    #         fip_image = glob(os.path.join(nnUNet_raw, opts.DName, 'labelsTr', im)+'*')[0]
    #         arr = CTImage(fip_image).image()
    #         for c in range(label_ignore):   
    #             weight[c] = weight[c] + (arr==c).sum()/arr.size
    #     weight = 1/weight
    #     weight[0] = 0
    #     classWeight = weight/weight.sum()
    #     return classWeight
        

    # def grad_data(self, opts, folderDict, man, net, data, layer_collection, idxG=None, idx_class=None, dict_grad={}):
        
    #     # idxG=None, idx_class=None, append=False, fisherG=True, use_mask=False, save_ent=False
        
    #     # data = data_train
    #     # idxG=[1,2,3,4,5]
    #     # use_mask=True
        
    #     def loss_fn(out, target):
    #         pred_weak_bin_log = torch.log_softmax(out, dim=1)
    #         pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
    #         loss=[]
    #         for c in range(pred_weak_bin_prop.shape[1]):
    #             loss.append(torch.mean(pred_weak_bin_log[:,c] * target[:,c]))
    #         loss = torch.mean(torch.stack(loss))
    #         return loss

    #     def compute_grad(sample, sample_target, opts, dict_grad):
    #         sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    #         pred = net.model['unet'].network(sample)[0]              
    #         pred_weak_bin_log = torch.log_softmax(pred, dim=1)
    #         pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
    #         if dict_grad['entropy']:
    #             dimSum = tuple([i for i in range(2,2+opts.dim)])
    #             entM = (-pred_weak_bin_log*pred_weak_bin_prop)
    #             dict_grad['entropy'] = entM.mean(dimSum).detach().cpu().numpy()
    #         else:
    #             dict_grad['entropy'] = None
                
    #         if dict_grad['entropy_map']:
    #             dict_grad['entropy_map'] = entM

    #         n_output = pred_weak_bin_prop.shape[1]
    #         if sample_target is None:
    #             #print('pred_weak_bin_prop123', pred_weak_bin_prop.shape)
    #             if opts.dim==2:
    #                 sample_target = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 3, 1, 2)
    #             else:
    #                 sample_target = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 4, 1, 2, 3)
    #         else:
    #             if opts.dim==2:
    #                 sample_target = torch.nn.functional.one_hot(sample_target,n_output).permute(0, 3, 1, 2)
    #             else:
    #                 #print('sample_target1234', sample_target.shape)
    #                 #print('sample_target12345', torch.unique(sample_target))
    #                 #print('n_output123', n_output)
    #                 sample_target = torch.nn.functional.one_hot(sample_target,n_output+1).permute(0, 4, 1, 2, 3)
                    
    #                 sample_target = torch.reshape(sample_target, (1, n_output+1, -1))
    #                 #print('sample_target12348', sample_target.shape)
    #                 #print('sample_targetxxx', sample_target[0,:,0:3])
    #                 #print('pred1234', pred.shape)
    #                 pred = torch.reshape(pred, (1, n_output, -1))
    #                 #print('sample_target12347', sample_target.shape)
    #                 idx = torch.where(sample_target[0,n_output,:]==0)[0]
    #                 #print('idx123', idx.shape)
    #                 #print('idx1234', idx.shape)
    #                 sample_target = sample_target[:,0:n_output,idx]
    #                 pred = pred[:,:,idx]
    #                 #print('sample_target123', sample_target.shape)
    #                 #print('pred123', pred.shape)
    #                 #sys.exit()
            
    #         #print('sample_target123', sample_target.shape)
    #         #plt.imshow(sample_target[0,1,:,:].detach().cpu().numpy())
    #         #plt.show()
            
    #         #print('pred123', pred.shape)
    #         #print('sample_target123', sample_target.shape)
            
    #         loss = loss_fn(pred, sample_target)
    #         #print('loss123', loss)
    #         grad_params = torch.autograd.grad(loss, list(net.model['unet'].network.parameters()), allow_unused=True)
    #         gradv = torch.hstack([torch.reshape(gr, (-1,)) for gr in grad_params if gr is not None])
    #         dict_grad['grad'] = gradv
    #         return dict_grad
        
    #     def compute_sample_grads(data, target, opts, dict_grad):
    #         """ manually process each sample with per sample gradient """
            
    #         sample_grads=[]
    #         entropies=[]
    #         entropies_map=[]
    #         for i in range(data.shape[0]):
    #             dict_grad_filled = dict_grad.copy()
    #             dict_grad_filled = compute_grad(data[i], target[i], opts, dict_grad_filled)
    #             sample_grads.append(dict_grad_filled['grad']) 
    #             entropies.append(dict_grad_filled['entropy']) 
    #             entropies_map.append(dict_grad_filled['entropy_map']) 
    #             del dict_grad_filled
    #         #entropies = [compute_grad(data[i], target[i], save_ent) for i in range(data.shape[0])]
            
    #         #sample_grads, entropies = [compute_grad(data[i], target[i], save_ent) for i in range(data.shape[0])]
    #         return sample_grads, entropies, entropies_map

    #     #batch_size=32
    #     batch_size = dict_grad['batch_size']
    #     net = man.load_model(opts, folderDict, previous=dict_grad['previous'])
    #     net.model['unet'].network.eval()
    #     dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data, batch_size=batch_size, single=True)
    #     NumBatches = math.ceil(len(data)/dataloader_train.data_loader.batch_size)
    #     device='cuda'
    #     label_ignore = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
        
    #     for b in tqdm(range(NumBatches), desc='Compute gradient'):
    #         batch = next(dataloader_train)
    #         datab = batch['data']
    #         if dict_grad['use_mask']:
    #             target=batch['target'][0].long().to(device)
    #             #print('target123', torch.unique(target))
    #             # !!! Please check if this really happens only for a few voxel that "ignore" labeled voxels are still in the patches of the training set (corner voxel)
    #             #target[target==label_ignore]=0
    #             # !!!
    #             #target[-1]=target[-2]
    #             #if 10 in list(batch['idx']):
    #             #    sys.exit()
                
    #         else:
    #             target=[None for i in range(datab.shape[0])]
    #         IDX = batch['idx'][:,0]
    #         #sys.exit()

    #         datab = datab.to(device, non_blocking=True)
    #         per_sample_grads, entropies, entropies_map = compute_sample_grads(datab, target, opts, dict_grad)
            
    #         # Filter gradients
    #         if idxG is not None:
    #             per_sample_grads=[v[idxG] for v in per_sample_grads]
            
    #         # Set gradient in data
    #         for i,ID in enumerate(list(IDX)):
    #             data[ID].F['grad']=per_sample_grads[i].detach().cpu().double()
    #             if entropies[i] is not None:
    #                 data[ID].F['uc']=entropies[i][0]
    #                 if entropies_map[i] is not False:
    #                     data[ID].F['uc_map']=entropies_map[i][0]
                
    #             # !!!
    #             #data[ID].F['out']=False
                
    #         # !!!
    #         #data[ID].F['out']=True


    # def getNetLayer(self, opts, man, folderDict, layer_used, previous):
    #     # Extract layers
    #     net = man.load_model(opts, folderDict, previous=previous)
    #     if layer_used=='mid':
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'encoder' in layer and 'convs' in layer and '1.conv' in layer:
    #                 mid_layer=layer
    #         layers=[mid_layer]
    #     elif layer_used=='decoder':
    #         layers=[]
    #         for layer, mod in net.model['unet'].network.named_modules():
    #         #for layer, mod in net.model['unet'].network.named_parameters():
    #             if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
    #                 layers.append(layer)
    #     elif layer_used=='encoder_decoder':
    #         layers=[]
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'encoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
    #                 layers.append(layer)
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
    #                 layers.append(layer)
    #     elif layer_used=='decoder_mid':
    #         layers=[]
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
    #                 layers.append(layer)
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'encoder' in layer and 'convs' in layer and '1.conv' in layer:
    #                 mid_layer=layer
    #         layers.append(mid_layer)
    #     elif layer_used=='last':
    #         for layer, mod in net.model['unet'].network.named_modules():
    #             if 'decoder' in layer and 'convs' in layer and '1.conv' in layer:
    #                 lastlayer = layer
    #         layers=[lastlayer]
    #     elif layer_used=='manual':
    #         layers=[]
    #         layers.append('decoder.stages.4.convs.1.conv')
    #         layers.append('decoder.stages.5.convs.1.conv') 
    #         layers.append('decoder.stages.6.convs.1.conv')
    #     elif layer_used=='decoder-1-2-3-4':
    #         layers=[]
    #         layers.append('decoder.stages.1.convs.1.conv')
    #         layers.append('decoder.stages.2.convs.1.conv') 
    #         layers.append('decoder.stages.3.convs.1.conv') 
    #         layers.append('decoder.stages.4.convs.1.conv') 
    #     elif layer_used=='layer6':
    #         layers=[]
    #         layers.append('decoder.stages.6.convs.1.conv') 
    #     layer_collection = LayerCollection.from_model_BF(net.model['unet'].network, ignore_unsupported_layers=True,layer_in=layers)
    #     return net, layer_collection, layers


    # def computeF(self, opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, idxG=None, idx_class=None, dict_grad={}):
        
    #     eps=1e-20

    #     # Compute gradients
    #     self.grad_data(opts, folderDict, man, net, data_F, layer_collection, idxG=idxG, idx_class=idx_class, dict_grad=dict_grad)
    #     params = layer_collection.get_parameters_BF(net.model['unet'].network)
    #     num_params = int(np.sum([x.numel() for x in params]))
    #     gradEmb = torch.zeros(data_F[0].F['grad'].shape)
    #     for s in data_F:
    #         gradEmb = gradEmb+(s.F['grad']*s.F['grad'])
    #     gradEmb = (gradEmb/len(data_F)).numpy().astype('float64')
    #     population = [x for x in range(len(gradEmb))]
    #     prop = gradEmb/gradEmb.sum()
    #     NumParamsSel = min(NumParams, (gradEmb>eps).sum())
    #     if idxG is None:
    #         idxG = np.random.choice(population, size=min(num_params, NumParamsSel), replace=False, p=prop)
    #         F=torch.from_numpy(gradEmb[idxG])
    #     else:
    #         F=torch.from_numpy(gradEmb)
    #     FI=(1/F).cuda()

    #     return FI, idxG


    # def getUncertainty(self, opts, folderDict, man, data_query, batch_size=None, save_uc=False, device='cuda', previous=True, NumMCDRounds=10, dropout_rate=0.01):

    #     # self=strategy
    #     # NumSamplesMax=15000
    #     # NumMCDRounds=10
    #     # save_uc=True
    #     # data_query=data_A
    #     # batch_size=8
    #     # device='cuda'
    #     # dropout_rate=0.1

    #     # init model
    #     net = man.load_model(opts, folderDict, previous=previous)
    #     net.model['unet'].network.eval()
    #     enable_dropout(net.model['unet'].network, dropout_rate)
        
    #     dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data_query, batch_size=batch_size)
    #     NumBatches = math.ceil(len(data_query)/dataloader_train.data_loader.batch_size)
    #     uc=[]
    #     for b in tqdm(range(NumBatches), desc='Estimate uncertainty'):
    #         batch = next(dataloader_train)
    #         IDX = batch['idx'][:,0]
    #         datab = batch['data']
    #         datab = datab.to(device, non_blocking=True)
            
    #         # Iterate over monte carlo rounds
    #         MCDrounds=[]
    #         for r in range(NumMCDRounds):
    #             out = net.model['unet'].network(datab)
    #             for i in range(len(out)): out[i] = out[i].detach_().cpu()
    #             outs = soft1(out[0])
    #             MCDrounds.append(torch.unsqueeze(outs, dim=0).clone())
    #         MCDrounds = torch.vstack(MCDrounds)
    #         ucMap = torch.var(MCDrounds, dim=0)
    #         #dimMean = tuple([i for i in range(2,2+opts.dim)])
    #         #uc = torch.mean(ucMap, dim=dimMean).numpy()
    #         dimSum = tuple([i for i in range(2,2+opts.dim)])
    #         uc = torch.sum(ucMap, dim=dimSum).numpy()
            
    #         # Set uncertainties in data
    #         for i,ID in enumerate(list(IDX)):
    #             data_query[ID].F['uc']=uc[i]
    #             if save_uc: 
    #                 data_query[ID].F['ucMap']=ucMap[i]

    # def getEntGrad(self, opts, folderDict, man, data_query, batch_size=None, save_uc=False, device='cuda', previous=True):

    #     # self=st
    #     # man=alunet.man
    #     # previous=False
    #     # data_query=man.datasets['query'].data
    #     # batch_size=2
    #     # device='cuda'
    #     # folderDict=alunet.man.folderDict
        
    #     NumParams=1000
    #     NumFIMMax=200
    #     NumSamplesMax=5000
    #     layer_used='decoder'
    #     net, layer_collection, _ = self.getNetLayer(opts, man, folderDict, layer_used, previous=True)
    #     net.model['unet'].network.eval()
        
    #     data = list(np.random.choice(data_query, size=min(NumSamplesMax, len(data_query)), replace=False))

    #     # # Filter parameters
    #     # layernames = ['.'.join(x.split('.')[0:-1]) for x in list(layer_collection.layers.keys())]
    #     # params={}
    #     # buffers = {}
    #     # for k, v in net.model['unet'].network.named_parameters():
    #     #     name = '.'.join(k.split('.')[0:-1])
    #     #     if name in layernames:
    #     #         params[k]=v
                
    #     # def loss_fn(out, target):
    #     #     pred_weak_bin_log = torch.log_softmax(out, dim=1)
    #     #     pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
    #     #     loss=[]
    #     #     for c in range(pred_weak_bin_prop.shape[1]):
    #     #         loss.append(torch.mean(pred_weak_bin_log[:,c] * target[:,c]))
    #     #     loss = torch.mean(torch.stack(loss))
    #     #     return loss
        
    #     # def compute_loss(params, buffers, sample, target):
    #     #     batch = sample.unsqueeze(0)
    #     #     targets = target.unsqueeze(0)
    #     #     predictions = functional_call(net.model['unet'].network, (params, buffers), (batch,))[0]
    #     #     loss = loss_fn(predictions, targets)
    #     #     return loss
        
    #     # data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False))
        
    #     # def compute_grad_ent(data, idx):
    #     #     # data=data_F
    #     #     dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data, batch_size=batch_size)
    #     #     NumBatches = math.ceil(len(data)/dataloader_train.data_loader.batch_size)
    #     #     entL=[]
    #     #     gradL=[]
    #     #     for b in tqdm(range(NumBatches), desc='Estimate uncertainty and gradient'):
    #     #         #sys.exit()
    #     #         batch = next(dataloader_train)
    #     #         IDX = batch['idx'][:,0]
    #     #         datab = batch['data']
    #     #         datab = datab.to(device, non_blocking=True)
                
    #     #         pred = net.model['unet'].network(datab)[0]
    #     #         pred_weak_bin_log = torch.log_softmax(pred, dim=1)
    #     #         pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
                
    #     #         bs = pred_weak_bin_prop.shape[0]
    #     #         n_output = pred_weak_bin_prop.shape[1]
    #     #         #targets = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 3, 1, 2)
    #     #         targets = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 4, 1, 2, 3)
                
                
    #     #         ft_compute_grad = grad(compute_loss)
    #     #         ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    #     #         ft_per_sample_grads = ft_compute_sample_grad(params, buffers, datab, targets)
                
    #     #         gradM = torch.hstack([ft_per_sample_grads[key].reshape(bs,-1) for key in ft_per_sample_grads])
    #     #         if idx is not None:
    #     #             gradM = gradM[:,idx]
    #     #         gradM = gradM.detach().cpu()
    #     #         gradL.append(gradM)
                
    #     #         #print('gradM', gradM.shape)
                    
    #     #         #ent = Categorical(probs = pred_weak_bin_prop.reshape(batch_size,2,-1)).entropy()
    #     #         ent = (-pred_weak_bin_log*pred_weak_bin_prop).mean((2,3)).detach().cpu().numpy()
    #     #         entL = entL + ent.tolist()
                
    #     #         # Set uncertainties in data
    #     #         for i,ID in enumerate(list(IDX)):
    #     #             data[ID].F['ent']=ent[i]
    #     #             data[ID].F['grad']=gradM[i].numpy()
              
    #     #     #print('gradL', len(gradL))
    #     #     gradL = torch.vstack(gradL)
    #     #     return gradL, entL
        
    #     # Compute fisher matrix
    #     data_F = list(np.random.choice(data, size=min(NumFIMMax, len(data)), replace=False))
    #     dict_grad = {'entropy': False, 'entropy_map': False, 'previous': True, 'use_mask': False, 'batch_size': 4}
    #     FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, dict_grad=dict_grad)
    #     data_query[0].F['FI']=FI
    #     data_query[0].F['idxG']=idxG
        
    #     dict_grad = {'entropy': True, 'entropy_map': False, 'previous': True, 'use_mask': False, 'batch_size': 4}
    #     self.grad_data(opts, folderDict, man, net, data, layer_collection, idxG=idxG, dict_grad=dict_grad)
        
    #     #dict_grad = {'entropy': False, 'entropy_map': False, 'previous': False, 'use_mask': False, 'batch_size': 4}
    #     #self.grad_data(opts, folderDict, man, net, data_pos, layer_collection, idxG=idxG, dict_grad=dict_grad)
        
    #     # for i,s in enumerate(data):
    #     #     print(i, s.F['uc'].sum())
        
        
        
    #     # alunet.load_data(opts, folderDict, alunet.man, data, batch_size=8, create_mask=False)
    #     # s = data[4]
    #     # s.showImageJ()
        
    #     # im = CTImage()
    #     # im.setImage(s.F['uc_map'][1,:,:,:].detach().cpu().numpy())
    #     # im.showImageJ()
        
    #     # im = CTImage()
    #     # im.setImage(s.F['uc_map'][1,:,:,:].detach().cpu().numpy())
    #     # im.showImageJ()
        
    #     # Ff = (gradF*gradF).mean(dim=0).cpu().numpy()
    #     # population = [x for x in range(len(Ff))]
    #     # prop = Ff/Ff.sum()
    #     # NumParamsSel = min(NumParams, (Ff>eps).sum())
    #     # idxG = np.random.choice(population, size=NumParamsSel, replace=False, p=prop)
    #     # F=torch.from_numpy(Ff[idxG])
    #     # FI=(1/F)
    #     # for s in data_F: 
    #     #     s.F['grad']=s.F['grad'][idxG]
        
    #     # # Compute gradient and entropy
    #     # gradQ, entQ = compute_grad_ent(data, idx=idxG)
        
    #     # # Set fsiher kernel for first sample
    #     # data[0].F['FI']=FI.numpy()
    #     # data[0].F['idxG']=FI.numpy()
 
        
