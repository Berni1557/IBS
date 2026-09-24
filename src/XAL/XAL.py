#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mount server: sudo mount -t cifs -o rw,user=foellmeb,uid=bernifoellmer,nounix,dir_mode=0777,file_mode=0777 //sc-data.sc-store.charite.de/sc-project-cc06-ag-dewey /mnt/hpc_XAL
# Folder access server: /sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/XAL
# Alocate gpu: srun -p gpu --pty -t 5:00:00 --mem=120G --gres=gpu:1 bash
# Access server: ssh foellmeb@s-sc-frontend1.charite.de


import os, sys
sys.path.append("/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src")
import shutil
from config.config import CConfig
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helper.DataAccess import DataAccess
from basemodel.DatasetBaseModel import YAML, YAML_MODE, defaultdict
from modules.XAL.ALSample import ALSample, ALSampleMulti
from modules.XAL.strategies.RandomStrategy import RandomStrategy
from modules.XAL.strategies.EntropyStrategy import EntropyStrategy, EntropyScanStrategy
from modules.XAL.strategies.MEANSTDStrategy import MEANSTDStrategy
from modules.XAL.strategies.UFALStrategy import UFALStrategy
from modules.XAL.strategies.UCORStrategy import UCORStrategy
from modules.XAL.strategies.BADGEStrategy import BADGEStrategy
from modules.XAL.strategies.UCBADGEStrategy import UCBADGEStrategy
from modules.XAL.ALManager import SALDataset

strategy_dict=({'RANDOM': RandomStrategy, 'ENTROPY': EntropyStrategy, 
                'ENTROPYSCAN': EntropyScanStrategy, 'MEANSTD': MEANSTDStrategy,
                'UFAL': UFALStrategy, 'UCOR': UCORStrategy, 'BADGE': BADGEStrategy,
                'UCBADGE': UCBADGEStrategy})
    
def create_dataset(opts):
    dataset = opts.CLDataset()
    hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    dataset.create_dataset_hdf5_seg(opts, hdf5filepath, NumSamples=None, test_set=True)


    from helper.DataAccess import DataAccess
    filepath = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACS/AL/hdf5_all.hdf5'
    # filepath = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACS/AL/INIT/INIT_V01/manager/train/data/sl.hdf5'
    hdf5filepath = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACS/AL/ENTROPY/ENTROPY_V02/manager/action/data/sl.hdf5'
    
    da = DataAccess()
    #hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    #hdf5filepath=manager.datasets['query'].fip_hdf5
    hdf5filepath = '/mnt/HHD/data/UCBADGE/LITS/AL/UCBADGE/UCBADGE_V02/manager/action/data/sl.hdf5'
    keys = da.read_keys(hdf5filepath)
    d=da.read_dict(hdf5filepath, ID=None, keys_select=['ID'])
    d=da.read_dict(hdf5filepath, ID=None, keys_select=['TRAIN', 'VALID', 'TEST'])
    
    # names = d['F']['imagename']
    
    # for i,n in enumerate(list(names)):
    #     if '.mhd' not in n:
    #         sys.exit()
    
    # import tables
    # file = tables.open_file(filepath, mode='r')
    # file.close()

def init_dataset(opts):

    # Init dataset
    method = 'INIT'
    NewVersion = True
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=method, NewVersion=NewVersion)
    dataset = opts.CLDataset()
    NumNewSamples = opts.AL_steps[0]
    #NumNewSamples = 100
    
    # manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    #manager.load(include=['action'], load_dict={'X': True, 'Y': True, 'F': True}, load_class=opts.CLSample, hdf5=True)

    # Init dataset
    manager.init_datasets(opts, folderDict, exclude=['labeled', 'unlabeled'])
    manager.save(save_dict=dict(), save_class=opts.CLSample, hdf5=True)
    
    # Create initial training set fro labeling
    fp_manual = opts_dict['fp_manual']
    action_round = manager.getRandom(dataset='query', NumSamples=NumNewSamples, remove=True)
    dataset.load_dataset_hdf5(folderDict, action_round)
    manager.datasets['action_round'].data = action_round
    #dataset.create_action(opts, folderDict, manager, fp_manual, action_round)
    

    ###### Automatic LABELING ######
    
    # # Update training set from labeled samples
    # action_round = manager.datasets['action_round'].data
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # dataset = opts.CLDataset()
    # dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)
    
    # Update datasets
    manager.datasets['action'].data = manager.datasets['action'].data + action_round
    manager.datasets['train'].data = manager.datasets['train'].data + action_round
    manager.datasets['query'].delete(action_round)
    manager.datasets['action_round'].delete(action_round)
    manager.save(include=['action'], save_dict={'X': True, 'Y': True, 'F': True}, save_class=opts.CLSample, hdf5=True)
    manager.save(include=['train', 'valid', 'query', 'test'], save_dict={}, save_class=opts.CLSample, hdf5=True)

    # Train model
    name_training = 'training_' + folderDict['name']
    NumSamplesTrainLoad = 1000
    NumSamplesValidLoad = 500
    hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    fip_hdf5_list = {'hdf5_all': hdf5_all, 
                     'train': manager.datasets['train'].fip_hdf5, 
                     'valid': manager.datasets['valid'].fip_hdf5,
                     'action': manager.datasets['action'].fip_hdf5,
                     'query': manager.datasets['query'].fip_hdf5,
                     'fip_action_previous': folderDict['fip_action_previous']}

    
    # manager.datasets['valid'].data = manager.datasets['train'].data
    
    manager.train(opts, 
                  name_training=name_training, 
                  fip_hdf5_list=fip_hdf5_list, 
                  grad_reg=False, 
                  settingsfilepath_tf=opts.settingsfilepath_tf, 
                  epochs=500, 
                  epoch_valid=5, 
                  NumSamplesTrainLoad=NumSamplesTrainLoad, 
                  NumSamplesValidLoad=NumSamplesValidLoad, 
                  savePretrained_all=opts.savePretrained_all,
                  lr=0.0001)



    # dataset = opts.CLDataset()

    # # Init dataset
    # method = 'INIT'
    # NewVersion = True
    # manager = opts.CLManager(fp_dataset=opts.dataset_data)
    # folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=method, NewVersion=NewVersion)
    # NumNewSamples = opts.AL_steps[0]
    # #NumNewSamples = 100
    
    # # manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    # #manager.load(include=['action'], load_dict={'X': True, 'Y': True, 'F': True}, load_class=opts.CLSample, hdf5=True)

    # # Init dataset
    # manager.init_datasets(opts, folderDict, exclude=['labeled', 'unlabeled'], TrainRatio=0.7, ValidRatio=0.10, NumPreSelect=None)
    # manager.save(save_dict=dict(), save_class=opts.CLSample, hdf5=True)
    
    # ###### MANUAL LABELING ######
    # fp_manual = opts_dict['fp_manual']
    # action_round = manager.getRandom(dataset='query', NumSamples=NumNewSamples, remove=True)
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # manager.datasets['action_round'].data = action_round
    # #manager.datasets['train'].data=[]
    # dataset.create_action(opts, folderDict, manager, fp_manual, action_round)
    # ###
    # action_round = manager.datasets['action_round'].data
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # dataset = opts.CLDataset()
    # dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)
    
    # # Update datasets
    # #manager.datasets['train'].data = manager.datasets['train'].data + manager.datasets['action'].data
    # manager.datasets['action'].data = manager.datasets['action'].data + action_round
    # manager.datasets['train'].data = manager.datasets['train'].data + action_round
    # manager.datasets['query'].delete(action_round)
    # manager.datasets['action_round'].delete(action_round)
    # manager.save(include=['action'], save_dict={'X': True, 'Y': True, 'F': True}, save_class=opts.CLSample, hdf5=True)
    # manager.save(include=['train', 'valid', 'query', 'test'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    
    # #############################
    
    # # Train model
    # name_training = 'training_' + folderDict['name']
    # NumSamplesTrainLoad = 1000
    # NumSamplesValidLoad = 500
    # hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    # fip_hdf5_list = {'hdf5_all': hdf5_all, 
    #                  'train': manager.datasets['train'].fip_hdf5, 
    #                  'valid': manager.datasets['valid'].fip_hdf5,
    #                  'action': manager.datasets['action'].fip_hdf5,
    #                  'query': manager.datasets['query'].fip_hdf5,
    #                  'fip_action_previous': folderDict['fip_action_previous']}

    
    # manager.train(opts, 
    #               name_training=name_training, 
    #               fip_hdf5_list=fip_hdf5_list, 
    #               grad_reg=False, 
    #               settingsfilepath_tf=opts.settingsfilepath_tf, 
    #               epochs=opts.epochs_init, 
    #               epoch_valid=5, 
    #               NumSamplesTrainLoad=NumSamplesTrainLoad, 
    #               NumSamplesValidLoad=NumSamplesValidLoad, 
    #               savePretrained_all=opts.savePretrained_all,
    #               lr=0.0001)

# def alround(opts):

#     NewVersion = True
#     VersionUse = None
#     strategy_name = opts.strategy
#     manager = opts.CLManager(fp_dataset=opts.dataset_data)
#     folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

#     # Load train
#     manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
#     version = folderDict['version']
#     NumSamples = opts.AL_steps[version]
#     manager.datasets['action'].data=[]
#     #NumSamples=3
    
#     #manager.datasets['query'].data =  manager.getRandom(dataset='query', NumSamples=200, remove=True)
#     data_query = manager.getRandom(dataset='query', NumSamples=20000, remove=True)
#     #data_query = manager.getRandom(dataset='query', NumSamples=300, remove=True)
    
#     # Sample new samples
#     CLStrategy = strategy_dict[strategy_name]
#     strategy = CLStrategy()
#     action_round = strategy.query(opts, folderDict, manager, data_query, opts.CLSample, NumSamples=NumSamples, batchsize=500, pred_class=['XRegionPred', 'XSegmentPred'], save_uc=True)
#     #action_round = strategy.query(opts, folderDict, manager, data_query, opts.CLSample, NumSamples=NumSamples, batchsize=500, pred_class=['XRegionPred', 'XSegmentPred'], save_uc=True)

#     ###############
#     # net = manager.load_model(opts, folderDict, previous=True)
#     # dataset = opts.CLDataset()
#     # dataset.load_dataset_hdf5(folderDict, action_round, debug=False)
#     # opts.CLSample.predict(action_round, net)
#     # for s in action_round:
#     #     s.plotSample(plotlist=['XImage', 'P'], color=True)
#     # # net = manager.load_model(opts, folderDict, previous=True)
#     # opts.CLSample.predict(action_round, net)
    
#     ###### test
#     #manager.load(include=['action', 'action_round'], load_class=opts.CLSample, hdf5=True)
#     # strategy = MEANSTDStrategy()
#     # batch = manager.datasets['action_round'].data
#     # dataset = opts.CLDataset()
#     # dataset.load_dataset_hdf5(folderDict, batch, debug=False)
#     # net = manager.load_model(opts, folderDict, previous=True)
#     # strategy.predict_uncertainty(opts, batch, net=net, NumMCD=10, pred_class='XSegmentPred', save_uc=True)
    
#     # net.disable_dropout() 
#     # opts.CLSample.predict(batch, net=net, eval_mode=True)

#     # idx=13
#     # batch[idx].plotSample(plotlist=['XImage', 'Y'])
#     # batch[idx].plotSample(plotlist=['P'], color=False)
#     # plt.imshow(batch[idx].info['uc_map'][0,0])
#     # plt.imshow(batch[idx].X['XImage'][0,2]>=130)
#     #####
    
#     # Manual labeling
#     #if not opts.emulation:
#     fp_manual = opts_dict['fp_manual']
#     dataset = opts.CLDataset()
#     dataset.load_dataset_hdf5(folderDict, action_round)
#     for s in action_round:
#         #s.Y['XMain']=s.Y['XMain']*0
#         s.Y['XRegion']=s.Y['XRegion']*0
#         s.Y['XSegment']=s.Y['XSegment']*0
        
#     manager.datasets['action_round'] = SALDataset('action_round')
#     manager.datasets['action_round'].data = action_round
#     manager.save(include=['action_round'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    

#     ###### MANUAL LABELING ######
#     dataset.create_action(opts, folderDict, manager, fp_manual, action_round)
#     print('Actions created!')
#     sys.exit()
#     ###
#     action_round = manager.datasets['action_round'].data
#     dataset.load_dataset_hdf5(folderDict, action_round)
#     dataset = opts.CLDataset()
#     fp_manual = opts_dict['fp_manual']
#     dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)
#     #############################

    
#     #manager.save(include=['action'], save_dict={'XImage': True, 'XRegion': True, 'XMask': True, 'XSegment': True}, save_class=opts.CLSample, hdf5=True)

#     # Update datasets
#     #manager.datasets['train'].data = manager.datasets['train'].data + manager.datasets['action'].data
#     manager.datasets['action'].data = manager.datasets['action'].data + action_round
#     manager.datasets['train'].data = manager.datasets['train'].data + action_round
#     manager.datasets['query'].delete(action_round)
#     manager.datasets['action_round'].delete(action_round)
#     manager.save(include=['action'], save_dict={'X': True, 'Y': True}, save_class=opts.CLSample, hdf5=True)
#     manager.save(include=['train', 'valid', 'query'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    
#     # Train model
#     name_training = 'training_' + folderDict['name']
#     NumSamplesTrainLoad = 800
#     NumSamplesValidLoad = 700
#     hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
#     fip_hdf5_list = {'hdf5_all': hdf5_all, 
#                      'train': manager.datasets['train'].fip_hdf5, 
#                      'valid': manager.datasets['valid'].fip_hdf5,
#                      'action': manager.datasets['action'].fip_hdf5,
#                      'query': manager.datasets['query'].fip_hdf5,
#                      'fip_action_previous': folderDict['fip_action_previous']}
    

#     manager.train_step(opts, 
#                        name_training=name_training, 
#                        fip_hdf5_list=fip_hdf5_list, 
#                        grad_reg=False, 
#                        settingsfilepath_tf=opts.settingsfilepath_tf, 
#                        epochs=opts.epochs_init, 
#                        epoch_valid=25, 
#                        NumSamplesTrainLoad=NumSamplesTrainLoad, 
#                        NumSamplesValidLoad=NumSamplesValidLoad, 
#                        savePretrained_all=opts.savePretrained_all,
#                        lr=0.000001)


#     # manager.load(include=['action_round'], load_dict={'X': True}, load_class=opts.CLSample, hdf5=True)
#     # dataset.load_dataset_hdf5(folderDict, [s], debug=False)
#     # manager.load(include=['train'], load_dict={'X': True}, load_class=opts.CLSample, hdf5=True)
    

def alquery(opts):

    NewVersion = True
    VersionUse = None
    strategy_name = opts.strategy
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    version = folderDict['version']
    NumSamples = opts.AL_steps[version]
    manager.datasets['action'].data=[]
    #NumSamples=10
    
    #manager.datasets['query'].data =  manager.getRandom(dataset='query', NumSamples=200, remove=True)
    #data_query = manager.getRandom(dataset='query', NumSamples=20000, remove=True)
    data_query = manager.getRandom(dataset='query', NumSamples=5000, remove=True)
    
    # Sample new samples
    CLStrategy = strategy_dict[strategy_name]
    strategy = CLStrategy()
    
    action_round = strategy.query(opts, folderDict, manager, data_query, opts.CLSample, NumSamples=NumSamples, batchsize=500, pred_class=['XMaskPred'], save_uc=False)
    #action_round = strategy.query(opts, folderDict, manager, data_query, opts.CLSample, NumSamples=NumSamples, batchsize=500, pred_class=['XRegionPred', 'XSegmentPred'], save_uc=True)
    
    ID=[]
    for s in action_round:
        ID.append(s.ID)

    ### !!!
    def tmp():
        dataset = opts.CLDataset()
        #action_round = manager.datasets['action_round'].data[0:100]
        dataset.load_dataset_hdf5(folderDict, action_round)
        net = manager.load_model(opts, folderDict, previous=True)
        strategy = UCBADGEStrategy()
        strategy.predict_uncertainty_load(opts, folderDict, manager, action_round, opts.CLSample, net, NumMCD=10, batchsize=100, pred_class=['XMaskPred'], save_uc=True)
        dataset.load_dataset_hdf5(folderDict, action_round)
        opts.CLSample.predict(action_round, net=net)
        ucall=0
        for s in action_round[0:20]:
            #s = action_round[3]
            s.plotSample(plotlist=['XImage', 'Y', 'P'], color=False)
            plt.imshow(s.info['uc_map'][0,0,:,:])
            plt.show()
            ucall = ucall+s.info['uc']
            
        ucall=0
        for s in action_round:
            ucall = ucall+s.info['uc']
        
        from modules.XAL.strategies.UFALStrategy import FisherAL
        from nngeometry.object import PMatKFAC, PMatDiag
        import math
        layers=['conv_up3.conv2','conv_up4.conv2','conv_up5.conv2','conv_up6.conv2',
        'conv_up7.conv2', 'conv_up8.conv2']
        settingsfilepath = os.path.join(opts.dataset_data, 'fisher.yml')
        loss_fisher = opts.CLSample.loss_fisher
        fisher = FisherAL(settingsfilepath, loss_fisher=loss_fisher, representation=PMatDiag)
        Ft, layer_collection = fisher.predictFIM(net, action_round, layers=layers, pred_class='XMaskPred', n_output=2)  
        
        loss_func = opts.CLSample.loss_fisher(net, idx_output=None)
        gradEmbedding = strategy.get_grad_embedding(net, action_round, loss_func, layers)
        import torch.nn.functional as f
        gradEmbedding = f.normalize(gradEmbedding, p=2, dim=1)
        for i,s in enumerate(action_round):
            gradEmbedding[i,:] = gradEmbedding[i,:]*s.info['uc']

        for i in range(10):
            s = (Ft.inverse().get_diag().cpu()*gradEmbedding[i]*gradEmbedding[i]).mean()
            s2 = (Ft.get_diag().cpu()*gradEmbedding[i]*gradEmbedding[i]).mean()
            uc = action_round[i].info['uc']
            l = math.sqrt((gradEmbedding[i,:]*gradEmbedding[i,:]).sum())
            print('i:', i)
            print('s:', s)
            print('s2:', s2)
            print('uc:', uc)
            print('l:', l)
            print('---')
            
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(gradEmbedding)
        
        import pandas as pd
        import seaborn as sns
        df = pd.DataFrame()
        #df["y"] = y
        df["comp1"] = X_embedded[:,0]
        df["comp2"] = X_embedded[:,1]
        df["label"] = [i for i in range(len(X_embedded))]
                
        plt.figure(figsize=(16,10))
        p1 = sns.scatterplot(
            x="comp1", y="comp2",
            #hue="y",
            palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=1.0
        )
        
        for line in range(0,df.shape[0]):
            p1.text(df.comp1[line]+0.01, df.comp2[line], 
                    df.label[line], horizontalalignment='left', 
                    size='medium', color='black', weight='semibold')


    dataset = opts.CLDataset()
    dataset.load_dataset_hdf5(folderDict, action_round, debug=True)
    net = manager.load_model(opts, folderDict, previous=True)
    opts.CLSample.predict(action_round, net=net, eval_mode=True)
    # for s in action_round:
    #     s.plotSample(plotlist=['XImage', 'Y', 'P'], color=False)

    # # Manual labeling
    # #if not opts.emulation:
    # fp_manual = opts_dict['fp_manual']
    # dataset = opts.CLDataset()
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # for s in action_round:
    #     #s.Y['XMain']=s.Y['XMain']*0
    #     s.Y['XMask']=s.Y['XMask']*0
    #     #s.Y['XSegment']=s.Y['XSegment']*0
        
    manager.datasets['action_round'] = SALDataset('action_round')
    manager.datasets['action_round'].data = action_round
    manager.save(include=['action_round'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    

    ####### MANUAL LABELING ######
    #dataset.create_action(opts, folderDict, manager, fp_manual, action_round)
    #print('Actions created!')

def alupdate_auto(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    version = folderDict['version']
    NumSamples = opts.AL_steps[version]
    manager.datasets['action'].data=[]
    
    ### !!!
    def tmp():
        dataset = opts.CLDataset()
        action_round = manager.datasets['action_round'].data[0:100]
        dataset.load_dataset_hdf5(folderDict, action_round)
        net = manager.load_model(opts, folderDict, previous=True)
        strategy = UCBADGEStrategy()
        strategy.predict_uncertainty_load(opts, folderDict, manager, action_round, opts.CLSample, net, NumMCD=10, batchsize=100, pred_class=['XMaskPred'], save_uc=True)
        dataset.load_dataset_hdf5(folderDict, action_round)
        opts.CLSample.predict(action_round, net=net)
        for s in action_round[0:20]:
            #s = action_round[3]
            s.plotSample(plotlist=['XImage', 'Y', 'P'], color=False)
            plt.imshow(s.info['uc_map'][0,0,:,:])
            plt.show()
            
        ucall=0
        for s in action_round:
            ucall = ucall+s.info['uc']
        
        from modules.XAL.strategies.UFALStrategy import FisherAL
        from nngeometry.object import PMatKFAC, PMatDiag
        import math
        layers=['conv_up3.conv2','conv_up4.conv2','conv_up5.conv2','conv_up6.conv2',
        'conv_up7.conv2', 'conv_up8.conv2']
        settingsfilepath = os.path.join(opts.dataset_data, 'fisher.yml')
        loss_fisher = opts.CLSample.loss_fisher
        fisher = FisherAL(settingsfilepath, loss_fisher=loss_fisher, representation=PMatDiag)
        Ft, layer_collection = fisher.predictFIM(net, action_round, layers=layers, pred_class='XMaskPred', n_output=2)  
        
        loss_func = opts.CLSample.loss_fisher(net, idx_output=None)
        gradEmbedding = strategy.get_grad_embedding(net, action_round, loss_func, layers)
        import torch.nn.functional as f
        gradEmbedding = f.normalize(gradEmbedding, p=2, dim=1)
        for i,s in enumerate(action_round):
            gradEmbedding[i,:] = gradEmbedding[i,:]*s.info['uc']

        for i in range(10):
            s = (Ft.inverse().get_diag().cpu()*gradEmbedding[i]*gradEmbedding[i]).mean()
            s2 = (Ft.get_diag().cpu()*gradEmbedding[i]*gradEmbedding[i]).mean()
            uc = action_round[i].info['uc']
            l = math.sqrt((gradEmbedding[i,:]*gradEmbedding[i,:]).sum())
            print('i:', i)
            print('s:', s)
            print('s2:', s2)
            print('uc:', uc)
            print('l:', l)
            print('---')
            
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(gradEmbedding)
        
        import pandas as pd
        import seaborn as sns
        df = pd.DataFrame()
        #df["y"] = y
        df["comp1"] = X_embedded[:,0]
        df["comp2"] = X_embedded[:,1]
        df["label"] = [i for i in range(len(X_embedded))]
                
        plt.figure(figsize=(16,10))
        p1 = sns.scatterplot(
            x="comp1", y="comp2",
            #hue="y",
            palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=1.0
        )
        
        for line in range(0,df.shape[0]):
            p1.text(df.comp1[line]+0.01, df.comp2[line], 
                    df.label[line], horizontalalignment='left', 
                    size='medium', color='black', weight='semibold')

    ###
    
    dataset = opts.CLDataset()
    action_round = manager.datasets['action_round'].data
    dataset.load_dataset_hdf5(folderDict, action_round)
    # dataset = opts.CLDataset()
    # fp_manual = opts_dict['fp_manual']
    # dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)

    # Update datasets
    #manager.datasets['train'].data = manager.datasets['train'].data + manager.datasets['action'].data
    manager.datasets['action'].data = manager.datasets['action'].data + action_round
    manager.datasets['train'].data = manager.datasets['train'].data + action_round
    manager.datasets['query'].delete(action_round)
    manager.datasets['action_round'].delete(action_round)
    manager.save(include=['action'], save_dict={'X': True, 'Y': True}, save_class=opts.CLSample, hdf5=True)
    manager.save(include=['train', 'valid', 'query'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    
    
def alupdate_manual(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    version = folderDict['version']
    NumSamples = opts.AL_steps[version]
    manager.datasets['action'].data=[]
    
    ### !!!
    # dataset = opts.CLDataset()
    # action_round = manager.datasets['action_round'].data
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # net = manager.load_model(opts, folderDict, previous=True)
    # strategy = MEANSTDStrategy()
    # strategy.predict_uncertainty_load(opts, folderDict, manager, action_round, opts.CLSample, net, NumMCD=10, batchsize=100, pred_class='XSegmentPred', save_uc=True)
    # s = action_round[0]
    # s['uc_map']
    
    
    ###
    
    dataset = opts.CLDataset()
    action_round = manager.datasets['action_round'].data
    dataset.load_dataset_hdf5(folderDict, action_round)
    dataset = opts.CLDataset()
    fp_manual = opts_dict['fp_manual']
    dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)

    # Update datasets
    #manager.datasets['train'].data = manager.datasets['train'].data + manager.datasets['action'].data
    manager.datasets['action'].data = manager.datasets['action'].data + action_round
    manager.datasets['train'].data = manager.datasets['train'].data + action_round
    manager.datasets['query'].delete(action_round)
    manager.datasets['action_round'].delete(action_round)
    manager.save(include=['action'], save_dict={'X': True, 'Y': True}, save_class=opts.CLSample, hdf5=True)
    manager.save(include=['train', 'valid', 'query'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    
    
def altrain(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    # version = folderDict['version']
    # NumSamples = opts.AL_steps[version]
    # manager.datasets['action'].data=[]
    
    # ###
    # dataset = opts.CLDataset()
    # action_round = manager.datasets['action_round'].data
    # dataset.load_dataset_hdf5(folderDict, action_round)
    # dataset = opts.CLDataset()
    # fp_manual = opts_dict['fp_manual']
    # dataset.update_samples_from_action(opts, folderDict, manager, fp_manual, action_round)
    # #############################

    
    # #manager.save(include=['action'], save_dict={'XImage': True, 'XRegion': True, 'XMask': True, 'XSegment': True}, save_class=opts.CLSample, hdf5=True)

    # # Update datasets
    # #manager.datasets['train'].data = manager.datasets['train'].data + manager.datasets['action'].data
    # manager.datasets['action'].data = manager.datasets['action'].data + action_round
    # manager.datasets['train'].data = manager.datasets['train'].data + action_round
    # manager.datasets['query'].delete(action_round)
    # manager.datasets['action_round'].delete(action_round)
    # manager.save(include=['action'], save_dict={'X': True, 'Y': True}, save_class=opts.CLSample, hdf5=True)
    # manager.save(include=['train', 'valid', 'query'], save_dict={}, save_class=opts.CLSample, hdf5=True)
    
    # # Train model
    # name_training = 'training_' + folderDict['name']
    # NumSamplesTrainLoad = 1000
    # NumSamplesValidLoad = 500
    # hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    # fip_hdf5_list = {'hdf5_all': hdf5_all, 
    #                   'train': manager.datasets['train'].fip_hdf5, 
    #                   'valid': manager.datasets['valid'].fip_hdf5,
    #                   'action': manager.datasets['action'].fip_hdf5,
    #                   'query': manager.datasets['query'].fip_hdf5,
    #                   'fip_action_previous': folderDict['fip_action_previous']}
    

    # manager.train_step(opts, 
    #                     name_training=name_training, 
    #                     fip_hdf5_list=fip_hdf5_list, 
    #                     grad_reg=False, 
    #                     settingsfilepath_tf=opts.settingsfilepath_tf, 
    #                     epochs=600,
    #                     epoch_valid=10, 
    #                     NumSamplesTrainLoad=NumSamplesTrainLoad, 
    #                     NumSamplesValidLoad=NumSamplesValidLoad, 
    #                     savePretrained_all=opts.savePretrained_all,
    #                     lr=0.00001)

    ######################
    # Train model
    name_training = 'training_' + folderDict['name']
    NumSamplesTrainLoad = 1000
    NumSamplesValidLoad = 500
    hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    fip_hdf5_list = {'hdf5_all': hdf5_all, 
                      'train': manager.datasets['train'].fip_hdf5, 
                      'valid': manager.datasets['valid'].fip_hdf5,
                      'action': manager.datasets['action'].fip_hdf5,
                      'query': manager.datasets['query'].fip_hdf5,
                      'fip_action_previous': folderDict['fip_action_previous']}

    
    # manager.datasets['valid'].data = manager.datasets['train'].data
    
    manager.train(opts, 
                  name_training=name_training, 
                  fip_hdf5_list=fip_hdf5_list, 
                  grad_reg=False, 
                  settingsfilepath_tf=opts.settingsfilepath_tf, 
                  epochs=3000, 
                  epoch_valid=5, 
                  NumSamplesTrainLoad=NumSamplesTrainLoad, 
                  NumSamplesValidLoad=NumSamplesValidLoad, 
                  savePretrained_all=opts.savePretrained_all,
                  lr=0.0001)
    
    ####################
    
    # # Train model
    # name_training = 'training_' + folderDict['name']
    # NumSamplesTrainLoad = 1000
    # NumSamplesValidLoad = 500
    # hdf5_all = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
    # fip_hdf5_list = {'hdf5_all': hdf5_all, 
    #                   'train': manager.datasets['train'].fip_hdf5, 
    #                   'valid': manager.datasets['valid'].fip_hdf5,
    #                   'action': manager.datasets['action'].fip_hdf5,
    #                   'query': manager.datasets['query'].fip_hdf5,
    #                   'fip_action_previous': folderDict['fip_action_previous']}
    

    # manager.train_step_shrink(opts, 
    #                     name_training=name_training, 
    #                     fip_hdf5_list=fip_hdf5_list, 
    #                     grad_reg=False, 
    #                     settingsfilepath_tf=opts.settingsfilepath_tf, 
    #                     epochs=600,
    #                     epoch_valid=10, 
    #                     NumSamplesTrainLoad=NumSamplesTrainLoad, 
    #                     NumSamplesValidLoad=NumSamplesValidLoad, 
    #                     savePretrained_all=opts.savePretrained_all,
    #                     lr=0.0001)



def altest(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    manager = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = manager.createALFolderpath(fp_active=opts.fp_active, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    manager.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLSample, hdf5=True)
    version = folderDict['version']
    NumSamples = opts.AL_steps[version]
    manager.datasets['action'].data=[]
    
    ### !!!
    dataset = opts.CLDataset()
    action_round = manager.datasets['action_round'].data
    dataset.load_dataset_hdf5(folderDict, action_round)
    net = manager.load_model(opts, folderDict, previous=True)
    strategy = MEANSTDStrategy()
    strategy.predict_uncertainty_load(opts, folderDict, manager, action_round, opts.CLSample, net, NumMCD=10, batchsize=100, pred_class='XSegmentPred', save_uc=True)
    opts.CLSample.predict(action_round, net=net)
    #s = action_round[0]
    #s['uc_map']
    
    
    
    idx=28
    action_round[idx].plotSample(plotlist=['XImage', 'P', 'Y'], color=True)
    plt.imshow(action_round[idx].info['uc_map'][0,0,:,:])
    for i in range(100):
        print('UC:', action_round[i].info['uc'])
        
    data_query = manager.getRandom(dataset='query', NumSamples=500, remove=True)
    strategy.predict_uncertainty_load(opts, folderDict, manager, data_query, opts.CLSample, net, NumMCD=10, batchsize=100, pred_class='XSegmentPred', save_uc=True)
    ent = np.array([x.info['uc'] for x in data_query])
    idx = np.argsort(ent)[::-1]
    samples = [data_query[i] for i in idx[0:NumSamples]]
    
    for i in range(100):
        print('UC:', samples[i].info['uc'])
    
    
    
    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name', type=str, default='LITS')
    parser.add_argument('--strategy', help='AL strategy', type=str, default='BADGE')
    parser.add_argument('--emulation', help='Function to execute', type=str, default=False)
    parser.add_argument('--func', help='Function to execute', type=str, default='')
    opts = parser.parse_args()
    opts_dict = vars(opts)

    # Define folder path
    if CConfig['hostname']=='foellmer':
        opts_dict['fp_active'] = os.path.join('/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data', opts.dataset, 'AL')
        opts_dict['dataset_data'] = os.path.join('/mnt/SSD2/cloud_data/Projects/CTP/src/modules', opts.dataset, 'data')
        opts_dict['settingsfilepath_tf'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/visualizer/TensorboardViewerTorch.yml'
        opts_dict['fp_manual'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/XALabeler/XALabeler/data_manual'
        #opts_dict['fp_images'] = '/home/bernhard/code/CTP/src/modules/SegmentCACS/data/images'
        #opts_dict['fp_references'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACSSeg/AL/data/references'
        opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
    else:
        opts_dict['fp_active'] = os.path.join('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/XAL/data', opts.dataset, 'AL')
        opts_dict['dataset_data'] = os.path.join('/sc-projects/sc-proj-cc06-ag-dewey/XAL', opts.dataset, 'data')
        opts_dict['settingsfilepath_tf'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/visualizer/TensorboardViewerTorch.yml'
        opts_dict['fp_manual'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/XAL/XALabeler/XALabeler/data_manual'
        #opts_dict['fp_images'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/SegmentCACS/data/images'
        #opts_dict['fp_references'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/XAL/data/SegmentCACSSeg/AL/data/references'
        opts_dict['fp_modules'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules'

    # Define classes
    if opts.dataset=='LITS':
        from modules.XAL.datasets.ALLITS import ALLITSManager, ALLITSDataset, ALLITSSample
        
        opts_dict['CLDataset'] = ALLITSDataset
        opts_dict['CLSample'] = ALLITSSample
        opts_dict['CLManager'] = ALLITSManager
        opts_dict['fp_images'] = '/mnt/HHD/data/LITS17/images'
        opts_dict['fp_references'] = '/mnt/HHD/data/LITS17/references'  
        opts_dict['fp_references_org'] = '/mnt/HHD/data/LITS17/references'  
        if CConfig['hostname']=='foellmer':
            #opts_dict['fp_active'] = os.path.join('/mnt/HHD/data/SegmentCACSSeg/XAL/data', opts.dataset, 'AL')
            opts_dict['fp_active'] = os.path.join('/mnt/HHD/data/UCBADGE', opts.dataset, 'AL')
        else:
            opts_dict['fp_images'] = '/mnt/HHD/data/LITS17/images'
            opts_dict['fp_references'] = '/mnt/HHD/data/LITS17/references'  
            opts_dict['fp_references_org'] = '/mnt/HHD/data/LITS17/references'  
            opts_dict['fp_active'] = os.path.join('/mnt/HHD/data/SegmentCACSSeg/XAL/data', opts.dataset, 'AL')
        
        # opts_dict['fp_images'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/SegmentCACS/data/images'
        # opts_dict['fp_references'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACSSeg/AL/data/references'        
        # opts_dict['fp_references_org'] = os.path.join(opts_dict['dataset_data'], 'references')
        # opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        # opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        
        opts_dict['AL_steps'] = [100] + [100 for i in range(20)]
        opts_dict['epochs_init'] = 500
        opts_dict['epochs_refine'] = 300
        opts_dict['epoch_valid'] = 15
        opts_dict['savePretrained_all'] = True
        
        # from modules.LITS.ALLiverSDataset import ALLiverSDataset, YAML, YAML_MODE, defaultdict
        # from modules.SpleenCNN.SpleenManager import SpleenManager
        # from modules.SpleenCNN.Sample_Spleen import Sample_Spleen
        # from modules.SpleenCNN.SpleenModel import SpleenModel
        # opts_dict['fp_images'] = '/mnt/HHD/data/LITS17/images'
        # opts_dict['fp_references_org'] = '/mnt/HHD/data/LITS17/references'
        # opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        # CLASSDataset = ALLiverSDataset
        # CLASSManager = SpleenManager
        # CLASSSample = Sample_Spleen
        # CLASSModel = SpleenModel
        # AL_steps = [200] + [100 for i in range(20)]
        # opts_dict['epochs_init'] = 500
        # opts_dict['epochs_refine'] = 300
        # opts_dict['epoch_valid'] = 15
        # opts_dict['savePretrained_all'] = True
        pass
    elif opts.dataset=='SegmentCACS':
        from modules.XAL.datasets.ALSegmentCACS import ALSegmentCACSManager, ALSegmentCACSDataset, ALSegmentCACSSample
        opts_dict['CLDataset'] = ALSegmentCACSDataset
        opts_dict['CLSample'] = ALSegmentCACSSample
        opts_dict['CLManager'] = ALSegmentCACSManager
        opts_dict['fp_images'] = os.path.join(opts_dict['dataset_data'], 'images')
        opts_dict['fp_references_org'] = os.path.join(opts_dict['dataset_data'], 'references')
        opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        opts_dict['AL_steps'] = [500] + [150 for i in range(20)]
        opts_dict['epochs_init'] = 500
        opts_dict['epochs_refine'] = 300
        opts_dict['epoch_valid'] = 15
        opts_dict['savePretrained_all'] = True
    elif opts.dataset=='SegmentCACSSeg':
        from modules.XAL.datasets.ALSegmentCACS import ALSegmentCACSManager, ALSegmentCACSDataset, ALSegmentCACSSample
        opts_dict['CLDataset'] = ALSegmentCACSDataset
        opts_dict['CLSample'] = ALSegmentCACSSample
        opts_dict['CLManager'] = ALSegmentCACSManager
        #opts_dict['fp_images'] = os.path.join(opts_dict['dataset_data'], 'images')
        #opts_dict['fp_images'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/SegmentCACS/data/images'
        #opts_dict['fp_references'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACSSeg/AL/data/references'        
        opts_dict['fp_references_org'] = os.path.join(opts_dict['dataset_data'], 'references')
        #opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        #opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        opts_dict['AL_steps'] = [100] + [100 for i in range(20)]
        opts_dict['epochs_init'] = 500
        opts_dict['epochs_refine'] = 300
        opts_dict['epoch_valid'] = 15
        opts_dict['savePretrained_all'] = True
    elif opts.dataset=='KITS':
        # from modules.KITS.KITSDataset import KITSDataset, YAML, YAML_MODE, defaultdict
        # from modules.SpleenCNN.SpleenManager import SpleenManager
        # from modules.SpleenCNN.Sample_Spleen import Sample_Spleen
        # from modules.SpleenCNN.SpleenModel import SpleenModel
        # opts_dict['fp_images'] = '/mnt/HHD/data/KiTS/kits19/data'
        # opts_dict['fp_references_org'] = '/mnt/HHD/data/KiTS/kits19/data'
        # opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        # CLASSDataset = KITSDataset
        # CLASSManager = SpleenManager
        # CLASSSample = Sample_Spleen
        # CLASSModel = SpleenModel
        # AL_steps = [200] + [100 for i in range(20)]
        # opts_dict['epochs_init'] = 500
        # opts_dict['epochs_refine'] = 300
        # opts_dict['epoch_valid'] = 15
        # opts_dict['savePretrained_all'] = True
        pass
    elif opts.dataset=='BRATS':
        # from modules.BRATS.BRATSDataset import BRATSDataset, YAML, YAML_MODE, defaultdict
        # from modules.BRATS.BRATSManager import BRATSManager
        # from modules.BRATS.Sample_BRATS import Sample_BRATS
        # from modules.BRATS.BRATSModel import BRATSModel
        # opts_dict['fp_images'] = '/mnt/HHD/data/BRATS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
        # opts_dict['fp_references_org'] = '/mnt/HHD/data/BRATS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
        # opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        # CLASSDataset = BRATSDataset
        # CLASSManager = BRATSManager
        # CLASSSample = Sample_BRATS  
        # CLASSModel = BRATSModel
        # AL_steps = [200] + [100 for i in range(20)]
        # opts_dict['epochs_init'] = 500
        # opts_dict['epochs_refine'] = 300
        # opts_dict['epoch_valid'] = 15
        # opts_dict['savePretrained_all'] = True
        pass
    else:
        raise ValueError('Dataset: ' + opts.dataset + ' does not exist.')
        
    # Start active learning
    if opts.func=='dataset_patches':      
        create_dataset(opts)
    elif opts.func=='dataset_init':      
        init_dataset(opts)
    elif opts.func=='alquery':      
        alquery(opts)
    elif opts.func=='alupdate_manual':      
        alupdate_manual(opts) 
    elif opts.func=='alupdate_auto':      
        alupdate_auto(opts) 
    elif opts.func=='altrain':      
        altrain(opts)      
        
    def comp_loop():
        for i in range(5):
            #alquery(opts)
            #alupdate_auto(opts) 
            altrain(opts)
        
        
        
    # import pandas as pd
    # import pyspssio
    # df = pd.read_spss('/mnt/SSD2/cloud_data/Projects/CTP/src/dischargedb/table1_and_2_09022022.sav')
    # df, meta = pyspssio.read_sav('/mnt/SSD2/cloud_data/Projects/CTP/src/dischargedb/tables/table1_and_2_09022022.sav')
