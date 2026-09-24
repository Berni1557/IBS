#!/usr/bin/env python3
"""Active-learning orchestration and nnU-Net data adapters.

This module contains the high-level ALUNET workflow, its dataset manager, and
the sample/patch types used to exchange data with nnU-Net.  Dataset-specific
conversion remains here because it is coupled to the experiment workflow.
"""

import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from collections import OrderedDict, defaultdict
from distutils.dir_util import copy_tree
from glob import glob
from os.path import join

import blosc2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
import torch
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    maybe_mkdir_p,
    save_json,
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import (
    nnUNetPredictor,
    compute_steps_for_sliding_window,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# Project modules
from ALUNET.ALUNetStrategies.CLASP import CLASP
from ALUNET.ALUNetStrategies.ENTROPY import ENTROPY
from ALUNET.ALUNetStrategies.IBS import IBS
from ALUNET.ALUNetStrategies.MCD import MCD
from ALUNET.ALUNetStrategies.RANDOM import RANDOM
from ALUNET.ALUNetStrategies.RANDOM66 import RANDOM66
from basemodel.DLBaseModel import DLBaseModel
from ct.ct import CTImage, CTRef
from helper.helper import compute_one_hot_torch, splitFilePath
from metrices.metrices import confusion, f1
from tools.nnUNet.nnUNet.nnunetv2.run.run_training import run_training
from XAL.ALAction import ALAction
from XAL.ALManager import ALManager, SALDataset
from XAL.ALSample import ALSample


STRATEGIES = {
    "RANDOM": RANDOM,
    "ENTROPY": ENTROPY,
    "MCD": MCD,
    "RANDOM66": RANDOM66,
    "CLASP": CLASP,
    "IBS": IBS,
}


def cuda_print():
    """Print a compact summary of CUDA availability and memory usage."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU:", torch.cuda.get_device_name(device))
        print("Allocated:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
        print("Reserved: ", torch.cuda.memory_reserved(device) / 1024**2, "MB")
        print("Max allocated:", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")
    else:
        print("No CUDA GPU available")


class ALUNET:
    """Coordinate dataset setup, annotation, querying, and model training."""

    def __init__(self, opts):
        opts_dict = vars(opts)
        opts_dict['CLDataset'] = UNetDatasetV2
        opts_dict['CLSample'] = UNetSampleV2
        opts_dict['CLPatch'] = UNetPatchV2
        opts_dict['CLManager'] = UNetManagerV2
        opts_dict['fp_images'] = ''
        opts_dict['fp_references'] = ''  
        opts_dict['fp_references_org'] = '' 
        opts_dict['nnUNetTrainer'] = opts.nnUNetTrainer 
        opts_dict['plans_identifier'] = 'ALUNETPlanner'
        opts_dict['DName'] = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        opts_dict['DNameFull'] = 'Dataset' + opts.dataset_name_or_id[0] + '01_' + opts.dataset

    def init_dataset(self, opts):
        """Initialize either the targeted or plain active-learning workflow."""
        if opts.targeted:
            self.init_dataset_target(opts)
        else:
            self.init_dataset_plain(opts) 

    def init_dataset_target(self, opts):
        
        # self=alunet
        
        # Init dataset
        method = 'INIT'
        NewVersion = False
        NumSamples = opts.ALSamples[0]
        NumSamplesPre = NumSamples*3
        self.man = opts.CLManager(fp_dataset='')
        opts.alunet = self
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=method, NewVersion=NewVersion)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
                
        # Set parameter
        #NumSamplesCheck = 50
        NumSamplesPre = NumSamples*3

        # Check if samples are unlabeled
        if opts.label_manual:
            if not folderDict['round_status']['correction_proposal']:
                # Convert to nnUNet structure
                self.convert_raw(opts, delete_label=True)
            
                # Preprocessing raw data
                subprocess.run(
                    [
                        "nnUNetv2_plan_and_preprocess",
                        "-d", opts.dataset_name_or_id,
                        "-pl", opts.plans_identifier,
                        "-overwrite_plans_name", opts.plans_identifier,
                        "-npfp", "1",
                        "-np", "1",
                        "--verify_dataset_integrity",
                        "-c", opts.configuration,
                        "--clean",
                    ],
                    check=True,
                )
                
                # Init nnUNet datasets
                self.man.init_patches(opts)
                self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
                
                # Label validation set
                if opts.label_valid:
                    self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)
                
                # Set correction flags
                self.man.update_status(folderDict, 'correction_proposal', True)
                self.man.update_status(folderDict, 'correction_selection_manual', True)
                self.man.update_status(folderDict, 'correction_preparation', True)
                self.man.update_status(folderDict, 'correction_annotation_manual', True)

            # Selection proposal
            if not folderDict['round_status']['selection_proposal']:
                # Query new samples
                strat = opts.strategy
                opts.strategy='RANDOM'
                self.query(opts, self.man.folderDict, self.man, NumSamples=NumSamplesPre, NumSamplesMaxMCD=5000)
                opts.strategy = strat
                self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
                self.man.datasets['action_round'].data=[]
                self.man.save(include=['action_round', 'action_pre'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                # Create action
                self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, create_mask=False, create_pseudo=False, pseudo_full=True, create_uc=False)
                #self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, create_mask=False, create_pseudo=True, pseudo_full=True)
                self.man.update_status(folderDict, 'selection_proposal', True)
            
    
            # Selection of targets (manual)
            if not folderDict['round_status']['selection_target_manual']:
                sys.exit('Please classifie images in positive and negative using XALabeler software!')
                # self.man.update_status(folderDict, 'selection_target_manual', True)
                    
            # Subset selection based on selected target samples
            if not self.man.folderDict['round_status']['selection_subset']:
                if not opts.interactive:
                    data_action_select = self.extract_data_action_select(opts, self.man.folderDict['fp_manual'])
                    data_select = self.select_manual(opts, self.man.folderDict, self.man, data_action_select, NumSamples=NumSamples)
                    self.man.datasets['query'].delete(data_select)
                    self.man.datasets['action_round'].data = data_select
                    self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                    # Create action
                    self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, classification_multislice=False, clustering=False, create_mask=False, create_pseudo=True)
                else:
                    data = self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data + self.man.datasets['query'].data 
                    fip_actionlist = os.path.join(self.man.folderDict['fp_manual'], 'actionlist.json')
                    actionlist = ALAction.load(fip_actionlist) 
                    action_pre = self.man.datasets['action_pre'].data
                    data_select = []
                    for a in actionlist:
                        for s in action_pre:
                            if a.id==s.ID:
                                s.action = a
                                a.filetype='.nii.gz'
                                if a.info['label']['batch']:
                                    data_select.append(s)
                    print('data_select123', len(data_select))
                    self.man.datasets['query'].delete(data_select)
                    self.man.datasets['action_round'].data = data_select
                    self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)
                
            # Annotated images (manual)
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                print('Doing selection_annotation_manual')
                if opts.segauto:
                    self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                    self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                else:
                    if not opts.interactive:
                        sys.exit('Please annotate images using XALabeler software!')
                        #_ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)
                        #self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                        #sys.exit('Please annotate images using XALabeler software!')
                        self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                    else:
                        self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                        
                    
        
            if not self.man.folderDict['round_status']['training']:

                # Label validation set
                if opts.label_valid:
                    self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)
                    
                if not opts.segauto:
                    _ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)

                # Train model
                self.train(opts)
                # Copy model and labeles
                self.copy_model_labels(opts, self.man.folderDict)
                self.man.update_status(self.man.folderDict, 'training', True)
              

        
    def init_dataset_plain(self, opts):
        
        # self=alunet
        
        # Init dataset
        method = 'INIT'
        NewVersion = False
        NumSamples = opts.ALSamples[0]
        self.man = opts.CLManager(fp_dataset='')
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=method, NewVersion=NewVersion)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
            
        # Check if samples are unlabeled
        if opts.label_manual:
            if not self.man.folderDict['round_status']['selection_subset']:
                # Convert to nnUNet structure
                self.convert_raw(opts, delete_label=True)
            
                # Preprocessing raw data
                #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                
                # Init nnUNet datasets
                self.man.init_patches(opts)
            
                # Init train set
                self.man.datasets['action_round'].data = self.man.getRandom(dataset='query', NumSamples=NumSamples, remove=True)
                self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, create_pseudo=False, create_uc=False)
                
                # Annotate validation set
                _ = self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)
                
                self.man.update_status(self.man.folderDict, 'correction_proposal', True)
                self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
                self.man.update_status(self.man.folderDict, 'correction_preparation', True)
                self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
                self.man.update_status(self.man.folderDict, 'selection_proposal', True)
                self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)
                
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                annotated = self.annotation_manual(opts, self.man, data_action=self.man.datasets['action_round'].data)
                if not annotated:
                    sys.exit('Please annotate images using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                #self.annotation_auto(opts, self.man, data=self.man.datasets['train'].data)

            if not self.man.folderDict['round_status']['training']:
                  # Preprocessing raw data
                  subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                
                  # Train model
                  self.train(opts)
                  # Copy model and labeles
                  self.copy_model_labels(opts, self.man.folderDict)
                  self.man.update_status(self.man.folderDict, 'training', True)
               
        else:

            # Convert to nnUNet structure
            self.convert_raw(opts, delete_label=True)
        
            # Preprocessing raw data
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)

            # Init nnUNet datasets
            self.man.init_patches(opts)
        
            # Init train set
            self.man.datasets['train'].data = self.man.getRandom(dataset='query', NumSamples=NumSamples, remove=True)
            self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
            
            # Update labels
            self.annotation_auto(opts, self.man, data=self.man.datasets['train'].data)
            self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)

            # Preprocessing raw data
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            
            # Train model
            self.train(opts)
            
            # Copy model and labeles
            self.copy_model_labels(opts, folderDict)
            
            # Update state
            self.man.update_status(self.man.folderDict, 'correction_proposal', True)
            self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
            self.man.update_status(self.man.folderDict, 'correction_preparation', True)
            self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
            self.man.update_status(self.man.folderDict, 'selection_subset', True)
            self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
            self.man.update_status(self.man.folderDict, 'training', True)


    def copy_model_labels(self, opts, folderDict):
        # Copy model back
        fp_results = os.path.join(opts.fp_nnunet, 'nnUNet_results', opts.DName, opts.nnUNetResults)
        fp_results_new = os.path.join(folderDict['modelpath'], opts.nnUNetResults)
        shutil.copytree(fp_results, fp_results_new, dirs_exist_ok=True)
        
        # Copy labels into model folder
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        fp_labelsTr_new = os.path.join(folderDict['modelpath'], 'labelsTr')
        shutil.copytree(fp_labelsTr, fp_labelsTr_new, dirs_exist_ok=True)
        

    def convert_raw(self, opts, delete_label=True):
        
        foldername = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        
        # setting up nnU-Net folders
        out_base = os.path.join(nnUNet_raw, foldername)
        imagestrTr = os.path.join(out_base, "imagesTr")
        labelstrTr = os.path.join(out_base, "labelsTr")
        imagestrTs = os.path.join(out_base, "imagesTe")
        labelstrTs = os.path.join(out_base, "labelsTe")
        maybe_mkdir_p(imagestrTr)
        maybe_mkdir_p(labelstrTr)
        maybe_mkdir_p(imagestrTs)
        maybe_mkdir_p(labelstrTs)
        
        if opts.dataset=='AMOS':
            label_ignore = 16
            # Copy imagesTr
            
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))[0:20]
            if not fip_images:
                expected_path = join(opts.fp_raw, opts.dataset, 'imagesTr')
                raise FileNotFoundError(
                    f"No AMOS training images found in {expected_path}. "
                    "Set raw_data_dir to the directory containing AMOS/imagesTr."
                )
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))[0:20]
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy labelTr
            
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))[0:20]
            print('fip_labels123', fip_labels)
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "spleen": 1,
                                      "right kidney": 2,
                                      "left kidney": 3,
                                      "gall bladder": 4,
                                      "esophagus": 5,
                                      "liver": 6,
                                      "stomach": 7,
                                      "arota": 8,
                                      "postcava": 9,
                                      "pancreas": 10,
                                      "right adrenal gland": 11,
                                      "left adrenal gland": 12,
                                      "duodenum": 13,
                                      "bladder": 14,
                                      "prostate/uterus": 15,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, 
                                  reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='KITS':
            label_ignore = 3
            # Copy imagesTr
            fip_cases = sorted(glob(join(opts.fp_raw, opts.dataset, 'dataset') + '/*'))[0:200]
            num_training_cases=len(fip_cases)
            for fip_case in tqdm(fip_cases, desc='Copy images'):
                fip_image = join(fip_case, 'imaging.nii.gz')
                im = CTImage(fip_image)
                arr = im.image()
                # Transform image
                arrs = arr.swapaxes(0, 2)
                ims = CTImage()
                ims.setImage(arrs)
                spacing = im.image_sitk.GetSpacing()
                ims.image_sitk.SetSpacing((spacing[2],spacing[1],spacing[0]))
                ims.image_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))

                if len(arrs.shape)==3:
                    ims.save(join(imagestrTr, os.path.basename(fip_case) + '_0000.nii.gz'))
                elif len(arrs.shape)==4:
                    for c in range(arrs.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arrs[c])
                        imc.SetOrigin(ims.GetOrigin())
                        imc.SetSpacing(ims.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_case) +'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # # Copy imagesTs
            num_test_cases = 0
            # Copy labelTr
            for fip_case in tqdm(fip_cases, desc='Copy labels'):
                fip_label = join(fip_case, 'segmentation.nii.gz')
                fip_image = join(imagestrTr, os.path.basename(fip_case) + '_0000.nii.gz')
                iml = sitk.ReadImage(fip_label)
                # Transform label
                arr = sitk.GetArrayFromImage(iml)
                arrs = arr.swapaxes(0, 2)
                imls = CTImage()
                imls.setImage(arrs)
                spacing = iml.GetSpacing()
                imls.image_sitk.SetSpacing((spacing[2],spacing[1],spacing[0]))
                imls.image_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
                # Save in data_raw/labeslTr
                fip_raw_labelsTr = join(opts.fp_raw, opts.dataset, 'labelsTr')
                sitk.WriteImage(imls.image_sitk, join(fip_raw_labelsTr, os.path.basename(fip_case) +'.nii.gz'), useCompression=True)

                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arrs = sitk.GetArrayFromImage(imls.image_sitk)
                    arrs[:] = label_ignore
                    imls = sitk.GetImageFromArray(arrs)
                    imls.SetOrigin(im.GetOrigin())
                    imls.SetSpacing(im.GetSpacing())
                    imls.SetDirection(im.GetDirection())
                else:
                    imls = imls
                sitk.WriteImage(imls, join(labelstrTr, os.path.basename(fip_case) +'.nii.gz'), useCompression=True)
        
            # # Copy labelTe
            # fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            # for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
            #     im = sitk.ReadImage(fip_label)
            #     sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "kidney": 1,
                                      "tumour": 2,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, 
                                  reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='ASOCA':
            label_ignore = 2
            # Copy imagesTr
            fip_images_normal = sorted(glob(join(opts.fp_raw, opts.dataset, 'Normal', 'CTCA') + '/*'))
            fip_images_diseased = sorted(glob(join(opts.fp_raw, opts.dataset, 'Diseased', 'CTCA') + '/*'))
            fip_images = fip_images_normal + fip_images_diseased
            num_training_cases = len(fip_images)
            print('fip_images_normal', join(opts.fp_raw, opts.dataset, 'Normal', 'CTCA'))
            #sys.exit('EXIT02')
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            num_test_cases = 0
                    
            # Copy labelTr
            fip_labels_normal = sorted(glob(join(opts.fp_raw, opts.dataset, 'Normal', 'Annotations') + '/*'))
            fip_labels_diseased = sorted(glob(join(opts.fp_raw, opts.dataset, 'Diseased', 'Annotations') + '/*'))
            fip_labels = fip_labels_normal + fip_labels_diseased
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                # Save in data_raw/labeslTr
                fip_raw_labelsTr = join(opts.fp_raw, opts.dataset, 'labelsTr')
                sitk.WriteImage(iml, join(fip_raw_labelsTr, os.path.basename(fip_label).split('.')[0] +'.nii.gz'), useCompression=True)

                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)

            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "artery": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, 
                                  reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)

        else:
            raise ValueError('Dataset conversation not implemented for dataset: ' + opts.dataset)
            
    
    def load_data(self, opts, folderDict, man, data, batch_size=8, create_pseudo=True, create_mask=True, previous=True, device='cuda'):
        # data=man.datasets['train'].data
        # Create dataloader
        load_weights=create_pseudo
        net = man.load_model(opts, folderDict, previous=previous, load_weights=load_weights)
        net.model['unet'].network.eval()
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data, batch_size=batch_size, single=True)
        NumBatches = math.ceil(len(data)/dataloader_train.data_loader.batch_size)
        
        NumClasses = len(load_json(os.path.join(nnUNet_raw, opts.DName, 'dataset.json'))['labels'])-1
        for b in tqdm(range(NumBatches), desc='Load patch'):
            batch = next(dataloader_train)
            datab = batch['data']
            IDX = batch['idx'][:,0]
            imagenames = batch['imagename']
            print('imagenames123', imagenames.shape)
            print('imagenames1234', imagenames)
            datab = datab.to(device, non_blocking=True)
            if create_pseudo:
                #print('datab123', datab)
                out = net.model['unet'].network(datab)
                #print('prediction123', (out[0][0,1,:,:]>0).sum())
                for i in range(len(out)): out[i] = out[i].detach_().cpu()
                pred = soft1(out[0])
            
            # Set prediction and mask 
            #fp_full_labels = join(nnUNet_preprocessed, opts.DNameFull, opts.plans_identifier+'_'+opts.configuration)
            fp_full_labels = join(nnUNet_preprocessed, opts.DNameFull, opts.plans_identifier+'_'+opts.configuration)
            for i,ID in enumerate(list(IDX)):
                if create_mask:
                    if i==0 or imagenames[i,0]!=imagenames[i-1,0]:
                        #fip = glob(join(fp_full_labels, imagenames[i,0]+'.npz'))[0]
                        #ref = CTRef(np.load(fip)['seg'][0])
                        
                        fip = join(fp_full_labels, imagenames[i,0]+'.b2nd')
                        print('fip123', fip)
                        arr = blosc2.open(fip)[0]
                        ref = CTRef()
                        ref.setRef(arr)
                        mask = ref.numTobin(NumClasses).ref()
                if create_pseudo:
                    data[ID].P['XMaskPred'] = compute_one_hot_torch(pred[i:i+1])
                    #del pred
                    #del out
                data[ID].X['XImage'] = datab[i:i+1]
                if create_mask:
                    data[ID].Y['XMask'] = dataloader_train.data_loader.generate_mask(data[ID], mask)
                    
            
    def query(self, opts, folderDict, man, NumSamples, batchsize=2, NumSamplesMaxMCD=5000):
        previous=True 
        strategy = STRATEGIES[opts.strategy]()
        self.man.alunet = self
        #action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=500, pred_class='XMaskPred', previous=previous, save_uc=False)
        action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=batchsize, pred_class='XMaskPred', previous=previous, save_uc=False, NumSamplesMaxMCD=NumSamplesMaxMCD)
        man.datasets['action_round'].data = action_round
        man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
    
    def select_manual(self, opts, folderDict, man, data_action, NumSamples):
        # self=alunet
        # man=self.man
        
        if 'IBS' in opts.strategy:
            strategy_name  = 'IBS'
        else:
            strategy_name = opts.strategy
        previous=True 
        strategy = STRATEGIES[strategy_name]()
        man.alunet = self
        data_select = strategy.select_manual2(opts, folderDict, man, man.datasets['query'].data, data_action, NumSamples)
        #man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        return data_select
        
    def annotation_auto(self, opts, man, data, full=False, update_train=True, update_query=True):
        # data=man.datasets['valid'].data
        # self.load_data(opts, folderDict, man, data, batch_size=8)
        dataSort,_ = UNetPatchV2.sort_F(data, prop='imagename')
        imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        file_ending = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['file_ending']
        for imn in tqdm(imagenames, desc='Auto annotation'):
            fip = os.path.join(fp_labelsTr, imn + file_ending)
            #print('fip123', fip)
            fip_raw = join(opts.fp_raw, opts.dataset, 'labelsTr', imn + file_ending)
            ref = CTRef(fip)
            arr = ref.ref()
            arr_raw = CTRef(fip_raw).ref()
            if full:
                arr = arr_raw
            else:
                for s in dataSort:
                    if s.F['imagename']==imn:
                        #print('s123', s.F['imagename'], s.F['lbs_org'])
                        lbs_org = s.F['lbs_org']
                        ubs_org = s.F['ubs_org']
                        arr[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]] = arr_raw[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]]
            ref.setRef(arr)
            ref.save(fip)
        
        # Update training set
        if update_train:
            man.datasets['query'].delete(man.datasets['action_round'].data)
            man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
            man.datasets['action_round'].data=[]
            man.save(include=['train','query','action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)

        #if update_query:
        #    self.man.datasets['query'].delete(data_select)

        return True

    def annotation_auto_correction(self, opts, man, data, full=False):
        # data=man.datasets['valid'].data
        # self.load_data(opts, folderDict, man, data, batch_size=8)
        dataSort,_ = UNetPatchV2.sort_F(data, prop='imagename')
        imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        file_ending = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['file_ending']
        for imn in tqdm(imagenames, desc='Auto annotation'):
            fip = os.path.join(fp_labelsTr, imn + file_ending)
            fip_raw = join(opts.fp_raw, opts.dataset, 'labelsTr', imn + file_ending)
            ref = CTRef(fip)
            arr = ref.ref()
            arr_raw = CTRef(fip_raw).ref()
            if full:
                arr = arr_raw
            else:
                for s in dataSort:
                    if s.F['imagename']==imn:
                        lbs_org = s.F['lbs_org']
                        ubs_org = s.F['ubs_org']
                        arr[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]] = arr_raw[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]]
            ref.setRef(arr)
            ref.save(fip)
        
        # Update training set
        #man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
        #man.datasets['action_round'].data=[]
        man.save(include=['train','action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)


    def create_action(self, opts, man, data, filetype='.nii.gz', classification=False, classification_multislice=False, clustering=False, show_roi=True, create_pseudo=True, create_uc=True, create_mask=True, pseude_is_mask=False, info_classified=False, info_selected=False, info_annotated=False, pseudo_full=True, strata=[]):
        # data=self.man.datasets['action_pre'].data
        # man=self.man
        # create_pseudo=False
        # create_mask=False
        # show_roi=True
        # classification=True
        # classification_multislice=True
        # pseudo_full=True
        # filetype='.nii.gz'
        
        
        folderDict = self.man.folderDict
    
       
        # Create folder structure
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
            
        # Delete fp_manual
        if os.path.isdir(fp_manual):
            shutil.rmtree(fp_manual, ignore_errors=True)
        os.makedirs(fp_manual, exist_ok=True)

        foldername = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        out_base = os.path.join(nnUNet_raw, foldername)
        labelstrTr = os.path.join(out_base, "labelsTr")
            
        fp_images = os.path.join(fp_manual, 'images')
        fp_pseudo = os.path.join(fp_manual, 'pseudo')
        fp_refine = os.path.join(fp_manual, 'refine')
        fp_mask = os.path.join(fp_manual, 'mask')
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        fip_color = os.path.join(fp_manual, 'XALabelerLUT.ctbl')
        os.makedirs(fp_pseudo, exist_ok=True)
        os.makedirs(fp_refine, exist_ok=True)
        os.makedirs(fp_mask, exist_ok=True)
        if create_uc:
            fp_uc = os.path.join(fp_manual, 'uc')
            os.makedirs(fp_uc, exist_ok=True)
        else:
            fp_uc = None
        
        labels = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']
        
        # Create colors
        colors_raw=[]
        if len(labels)>24:
            colors_raw_arr = np.round(np.random.uniform(low=0.0, high=1.0, size=(len(labels)+3,3)), decimals=1)
            for i in range(colors_raw_arr.shape[0]):
                colors_raw.append([colors_raw_arr[i,0], colors_raw_arr[i,1], colors_raw_arr[i,2]])
        else:
            colors_raw = [[0,0,0], [1,0,0], [0,0,1], [1,1,0], 
                          [0,1,1], [1,0,1], [0.75,0.5,0],[0.5,0.5,0.5],
                          [0.5,0,0], [0.5,0.5,0], [0,0.5,0], [0.5,0,0.5],
                          [0,0.5,0.5], [0,0,0.5], [1,1,1], [0.9,0.5,0.2], [0.2,0.5,0.2], [0.2,0.5,0.9],
                          [0.2,0.2,0.2], [0.5,0.5,0.2], [0.3,0.4,0.2], [0.7,0.2,0.8], [0.2,0.7,0.8],
                          [0.7,0.2,0.8]]
        
        
        

        if classification:
            labels['pseudo']=labels['ignore']+1
        else:
            labels['pseudo']=labels['ignore']+1
            
        colors=[]
        # for i in range(len(labels)-1):
        #     key = list(labels.keys())[list(labels.values()).index(i)]
        #     colors.append([i, key, colors_raw[i]])
        for i in range(len(labels)):
            key = list(labels.keys())[list(labels.values()).index(i)]
            if key =='ignore':
                colors.append([i, key, [0,1,0]])
            else:
                colors.append([i, key, colors_raw[i]])                
        lines = ['# Color']
        for i in range(len(colors)):
            if colors[i][1]=='ignore':
                trans = '0'
            else:
                trans = '255'
            col = str(colors[i][0]) + ' ' + str(colors[i][1]) + ' ' + str(colors[i][2][0]*255) + ' ' + str(colors[i][2][1]*255) + ' ' + str(colors[i][2][2]*255) + ' ' + trans
            lines = lines + [col]
        with open(fip_color, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n') 

        batch_size = opts.ALSamples[self.man.folderDict['version']-2]

        # Create settings file
        settings={'method': 'xalabeler',
                  'fp_images': os.path.join(nnUNet_raw, opts.DName, 'imagesTr'),
                  'fp_pseudo': fp_pseudo,
                  'fp_refine': fp_refine,
                  'fp_mask': fp_mask,
                  'fp_uc': fp_uc,
                  'fip_actionlist': fip_actionlist,
                  'fip_colors': fip_color,
                  'foregroundOpacity': 0.3,
                  "classification": classification,
                  "classification_multislice": classification_multislice,
                  "clustering": clustering,
                  "show_roi": show_roi,
                  "fip_round_status": man.folderDict['fip_round_status'],
                  "default_ignore": False,
                  "batch_size": batch_size}
        
        fip_settings = os.path.join(fp_manual, 'settings_XALabeler.json')
        with open(fip_settings, 'w') as file:
            file.write(json.dumps(settings, indent=4))
            
        # Create action list
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        
        # Sort by imagename
        dataSort, idx_sort = UNetPatchV2.sort_F(data, prop='imagename', prop_sorted=False)

        if create_uc:
            ucprop = UCPROP()
            previous = True
            ucprop.getUncertainty(opts, folderDict, man, data, batch_size=2, save_uc=True, device='cuda', previous=previous, NumMCDRounds=5, dropout_rate=0.1)
            UNetPatchV2.saveUC(opts, data, folderDict, fp_uc, filetype)

        if 'class' in strata:
            opts.alunet.load_data(opts, folderDict, man, data, batch_size=2, create_pseudo=True, create_mask=False)
            for s in data:
                s.F['strata_info']=[]
                for c in range(s.P['XMaskPred'].shape[1]):
                    if s.P['XMaskPred'][0,c,:,:].max()==1:
                        key = colors[c][1]
                        s.F['strata_info'].append(key)

        # Create actionlist
        actionlist=[]
        for s in dataSort:
            action=ALAction()
            action.name='action'
            action.status='open'
            action.id=s.F['ID']
            action.bboxLbsOrg=[x for x in list(s.F['lbs_org'])]
            action.bboxUbsOrg=[x for x in list(s.F['ubs_org'])]
            action.dim=opts.dim
            action.imagename=s.F['imagename']+'_0000' + filetype
            action.pseudoname=s.F['imagename'] + filetype
            action.refinename=s.F['imagename'] + filetype
            action.ucname=s.F['imagename'] + filetype
            action.filetype=filetype
            action.maskname=None
            action.label=colors
            action.info['class']=[]
            action.info['classified']=info_classified
            #action.info['selected']=info_selected
            action.info['annotated']=info_annotated
            if 'uc' in s.F:
                action.info['uc']=float(s.F['uc'].sum())
            if 'fg' in s.F:
                print('fg123', s.F['fg'].tolist())
                action.info['fg']=s.F['fg'].tolist()
            if 'class' in strata:
                action.info['strata_info']=s.F['strata_info']
            actionlist.append(action)
        ALAction.save(fip_actionlist, actionlist)
        
        if create_pseudo:
            if not pseudo_full:
                self.load_data(opts, folderDict, man, data, batch_size=2, create_pseudo=create_pseudo, create_mask=create_mask)
            opts.CLSample.savePseudo(opts, data, folderDict, fp_pseudo, filetype, pseudo_full=pseudo_full, tile_step_size=0.7)
            #self.predict_pseudo_from_files(opts, data, folderDict, fp_pseudo, filetype, pseudo_full=True, tile_step_size=0.9)
        
        if pseude_is_mask:
            shutil.rmtree(fp_pseudo)
            shutil.copytree(labelstrTr, fp_pseudo, dirs_exist_ok=True)
            #fip_labels_mask = glob(labelstrTr + '/*')
            #for fip_label in fip_labels_mask:
        return idx_sort
        
    def predict_pseudo_from_files(self, opts, data, folderDict, fp_pseudo, filetype, pseudo_full=False, tile_step_size=0.5):
        
        dataSort,_ = UNetPatchV2.sort_F(data, prop='imagename')
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        imagenames = list([s.F['imagename'] for s in dataSort])
        
        imagenames = list(np.unique(imagenames))
        fip_images=[]
        fip_pseudos=[]
        for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
            fip_image = glob(fp_imagesTr + '/'+imn+'_*')[0]
            print('fip_image123', fip_image)
            fip_pseudo = os.path.join(fp_pseudo, imn + filetype)
            modelpath = opts.alunet.man.folderDict['modelpath_prev']
            #opts.alunet.predict_image(opts, folderDict, modelpath, fip_image, fip_pseudo)
            fip_images.append(fip_image)
            fip_pseudos.append(fip_pseudo)
        
        # Create predictor
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
            )
        
        predictor.initialize_from_trained_model_folder(
            os.path.join(modelpath, opts.nnUNetResults),
            use_folds=(opts.fold, ),
            checkpoint_name='checkpoint_final.pth',
        )

        # Predict image
        #image = CTImage(fip_image)
        #img, props = SimpleITKIO().read_images([fip_image])
        print('fp_imagesTr123', fp_imagesTr)
        print('fp_pseudo123', fp_pseudo)
        ret = predictor.predict_from_files(fp_imagesTr, fp_pseudo)
        
        
        
    def predict_image(self, opts, folderDict, modelpath, fip_image, fip_predict, tile_step_size=0.5):
        
        # self=alunet
        # Create predictor
        predictor = nnUNetPredictor(
            #tile_step_size=0.5,
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
            )
        
        predictor.initialize_from_trained_model_folder(
            os.path.join(modelpath, opts.nnUNetResults),
            use_folds=(opts.fold, ),
            checkpoint_name='checkpoint_final.pth',
        )

        # Predict image
        image = CTImage(fip_image)
        img, props = SimpleITKIO().read_images([fip_image])
        ret = predictor.predict_single_npy_array(img, props, None, None, False)
        pred = CTRef(ret)
        pred.copyInformationFrom(image)
        pred.save(fip_predict)
        del image
        del pred
        
    

    def annotation_manual(self, opts, man, data_action, bg_offset=False, ignore_pseudo=False):
        
        # data_action=self.man.datasets['action_round'].data
        # man=alunet.man
        # bg_offset=False
        
        if len(data_action)==0:
            return None
        
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
            
        #print('fp_manual123', fp_manual)
            
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        # !!! in load function
        if not os.path.isfile(fip_actionlist):
            sys.exit('Could not find actionlist.json.')
        actionlist = ALAction.load(fip_actionlist) 
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        
        labels = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']
        label_ignore = labels['ignore']
        
        #print('actionlist123', actionlist)
        if ignore_pseudo:
            label_offset = labels['ignore']
        else:
            label_offset = 0
            
        # !!!
        #for s in data_action:
        #    print('IDS', s.F['ID'])
        #sys.exit('TEST2')

        #fip_refine = None
        annotated=True
        pbar = tqdm(total=len(data_action))
        pbar.set_description("Update action" )
        for a in actionlist:
            #print('a123', a)
            # !!!
            #if not hasattr(a, 'filetype'):
            a.filetype='.nii.gz'
            
            pbar.update()
            if a.status=='solved':
                #sys.exit()
                fip_refine = join(fp_manual, 'refine', a.pseudoname)
                s = opts.CLSample.getSampleByID(data_action, a.id)
                ref_refine = CTRef(fip_refine)
                arrR = ref_refine.ref()
                if bg_offset:
                    arrR = arrR-1
                    arrR[arrR==-1]=label_offset
                fip_label = os.path.join(fp_labelsTr, s.F['imagename']+a.filetype)
                ref_label = CTRef(fip_label)

                print('fip_label123', fip_label)
                print('fip_refine123', fip_refine)

                arrL = ref_label.ref()
                #arrLT = arrL[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]
                #arrRT = arrR[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]
                #arrBin = (arrRT!=label_ignore)*1
                #arrLT = (1-arrBin) * arrLT + arrBin * arrRT
                #arrL[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]] = arrLT
                
                arrBin = (arrR!=label_ignore)*1
                arrL = (1-arrBin) * arrL + arrBin * arrR
                
                
                ref_label.setRef(arrL)
                ref_label.save(fip_label)
                print('fip_label123', fip_label)
            else:
                annotated=False
        pbar.close()
        
        # Update training and query set
        if annotated:
            man.datasets['query'].delete(man.datasets['action_round'].data)
            man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
            man.datasets['action_round'].data=[]
            man.save(include=['train','query', 'action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            

        #sys.exit()
        
        return annotated
        
    def train(self, opts, copy_model=True, preprocess=False):
              
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        
        
        #name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=False)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Preprocessing raw data
        if preprocess:
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name ALUNETPlanner" +  " -pl ALUNETPlanner"  + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            
        
        print('opts.nnUNetTrainer123', opts.nnUNetTrainer)
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                      configuration=opts.configuration, 
                      fold=opts.fold,
                      trainer_class_name=opts.nnUNetTrainer, 
                      plans_identifier=opts.plans_identifier, 
                      pretrained_weights=None,
                      num_gpus=1, 
                      #use_compressed_data=False, 
                      export_validation_probabilities=False, 
                      continue_training=False, 
                      only_run_validation=False, 
                      disable_checkpointing=False, 
                      val_with_best=False,
                      device=torch.device('cuda'))   

        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
            
    def trainval(self, opts, copy_model=True, modelpath='modelpath_prev'):
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold,
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=os.path.join(folderDict[modelpath], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth'),
                     num_gpus=1, 
                     use_compressed_data=False, 
                     export_validation_probabilities=True, 
                     continue_training=False, 
                     only_run_validation=False, 
                     disable_checkpointing=False, 
                     val_with_best=True,
                     device=torch.device('cuda'))  
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)

    def test(self, opts, modelpath='modelpath_prev'):
        
        # self=alunet
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Predict train images
        fp_test = join(os.path.dirname(folderDict['fp_manager']), 'data', 'test')
        os.makedirs(fp_test, exist_ok=True)
        #modelpath='modelpath'
        #fip_image = '/mnt/HHD/data/UNetAL/data_raw/CTA17C/imagesTr/05-COP-0031_1.2.392.200036.9116.2.2424156352.1455761338.8.1264100003.1.nii.gz'
        #fip_predict = '/mnt/HHD/data/UNetAL/nnunet/nnUNet_results/Dataset604_CTA17C/nnUNetTrainer_ALUNET__nnUNetPlans__2d/fold_0/validation/image.nii.gz'
        modelpath = self.man.folderDict['modelpath_prev']
        fip_images = glob(join(nnUNet_raw, opts.DName, 'imagesTr/*'))
        
        # Filter masks
        fp_masks = '/mnt/HHD/data/CTAAutoplaque/data/surenjav/CADMAN/export/masks'
        fp_masks_pat = glob(fp_masks + '/*')
        
        patients = [os.path.basename(fip).split('_')[0] for fip in fip_images if os.path.basename(fip).split('_')[0] in [os.path.basename(fipm) for fipm in fp_masks_pat]]
        fip_images = sorted([fip for fip in fip_images if os.path.basename(fip).split('_')[0] in patients])
        fp_masks = sorted([fip for fip in fp_masks_pat if os.path.basename(fip) in patients])
        NumClasses=19
        
        fip_export = join(fp_test, 'perform.pkl')

        perform={'images': []}
        for fip_image, fp_mask in tqdm(zip(fip_images, fp_masks), desc='Evaluate prediction'):
            fip_predict = join(fp_test, os.path.basename(fip_image))
            fip_mask = join(fp_mask, 'segment_masks.mhd')
            self.predict_image(opts, folderDict, modelpath, fip_image, fip_predict)
            
            ref_pred = CTRef(fip_predict)
            ref_mask = CTRef(fip_mask)
            pred = ref_pred.numTobin(NumClasses, offset=0).ref().swapaxes(1, 3).reshape(-1, NumClasses).astype(np.int16)
            mask = ref_mask.numTobin(NumClasses, offset=0).ref().swapaxes(1, 3).reshape(-1, NumClasses).astype(np.int16)
            #ref_mask = CTRef(fip_mask).
            
            # Compute performance per image
            C = confusion(torch.from_numpy(mask), torch.from_numpy(pred), num_classes=NumClasses, device='cpu').numpy()
            p=dict()
            p['C'] = C
            p['F1'] = f1(C)
            p['F1_micro'] = f1(C[1:,1:], mode='micro')
            p['F1_bin'] = f1(C, binary_class=None)
            
            # Compute overall performance
            perform['images'].append(p)
            if 'C' not in perform:
                perform['C'] = C
            else:
                perform['C'] = perform['C'] + C
            perform['F1'] = f1(perform['C'])
            perform['F1_micro'] = f1(perform['C'][1:,1:], mode='micro')
            perform['F1_bin'] = f1(perform['C'], binary_class=None)
            
            # Save performance
            pickle.dump(perform, open(fip_export, 'wb'))
            
        # Read performance
        with open(fip_export, 'rb') as f:
            perform = pickle.load(f)

        
    def val(self, opts, copy_model=True, modelpath='modelpath_prev', val_with_best=False):

        
        # self=alunet
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold,
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=os.path.join(folderDict[modelpath], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth'),
                     num_gpus=1, 
                     #use_compressed_data=False, 
                     #export_validation_probabilities=True, 
                     export_validation_probabilities=False, 
                     continue_training=False, 
                     only_run_validation=True, 
                     disable_checkpointing=False, 
                     val_with_best=val_with_best,
                     device=torch.device('cuda'))  
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
            
    def train_step(self, opts, copy_model=True, modelpath='modelpath_prev', pretrained=True):
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        if pretrained:
            pretrained_weights = os.path.join(folderDict['modelpath_prev'], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth')
        else:
            pretrained_weights = None

        print('folderDict123_OK', folderDict)
        
            
        #print('pretrained_weights123', pretrained_weights)
        #sys.exit()
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold, 
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=pretrained_weights,
                     #pretrained_weights=None,
                     num_gpus=1, 
                     #use_compressed_data=False, 
                     export_validation_probabilities=False, 
                     continue_training=False, 
                     #continue_training=True, 
                     only_run_validation=False, 
                     disable_checkpointing=False, 
                     val_with_best=False,
                 device=torch.device('cuda'))
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
        
    def alload(self, opts, datatset='train'):
        # self=alunet
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=False)
        man.load(include=['train'], load_class=opts.CLPatch, hdf5=False)
        self.load_data(opts, folderDict, man, data=man.datasets[datatset].data)
        
        # for s in man.datasets['train'].data:
        #     s.plotSample(plotlist=['XImage', 'P', 'Y'], color=True)
        
    def alcreate(self, opts, copy_nnUNet_data=False):
        NewVersion = True
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        self.man.folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        for key in self.man.folderDict['round_status']: self.man.folderDict['round_status'][key]=False
        return man, self.man.folderDict

    def alquerytest(self, opts):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        previous=True 
        strategy = strategies[opts.strategy]()
        man.alunet = self
        #action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=500, pred_class='XMaskPred', previous=previous, save_uc=False)

    def alround(self, opts, copy_nnUNet_data=True):
        if opts.targeted:
            self.alround_target(opts, copy_nnUNet_data)
        else:
            self.alround_plain(opts, copy_nnUNet_data)
            
    
    def alround_plain(self, opts, copy_nnUNet_data=True):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        # copy_nnUNet_data=False
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        _ = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _= self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        opts.alunet = self
        
        if opts.label_manual:
            processed = np.all([self.man.folderDict['round_status'][key] for key in self.man.folderDict['round_status']])
            if processed:
                # Create new AL round  
                _, _ = self.alcreate(opts, copy_nnUNet_data=True)
                
            # Correction
            if opts.correction:
                pass
            else:
                self.man.update_status(self.man.folderDict, 'correction_proposal', True)
                self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
                self.man.update_status(self.man.folderDict, 'correction_preparation', True)
                self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
            
            # Set True
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
                
            if not self.man.folderDict['round_status']['selection_subset']:
                print('Doing selection_subset')
                # Query new samples
                self.query(opts, self.man.folderDict, self.man, NumSamples=opts.ALSamples[self.man.folderDict['version']-2])
                # Cre/mnt/hpc_XAL/data/UNetAL/CTA18/AL/tmp/USIMFTRCA_V09/maate action
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, create_pseudo=True, create_mask=False,info_classified=True, info_selected=True, info_annotated=False)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)
                sys.exit('Please annotate images using XALabeler software!')
                
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                print('Doing selection_annotation_manual')
                sys.exit('Please annotate images using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                
            if not self.man.folderDict['round_status']['training']:
                print('Doing training')
                _ = self.annotation_manual(opts, man=self.man, data_action=self.man.datasets['action_round'].data)
                # Preprocessing raw data
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                # Train model
                self.train_step(opts, self.man.folderDict, pretrained=True)

                # Copy model and labeles
                self.copy_model_labels(opts, self.man.folderDict)
                self.man.update_status(self.man.folderDict, 'training', True)
        else:
            # Create new AL round  
            self.alcreate(opts, copy_nnUNet_data=True)
            # Query new samples
            NumSamples=opts.ALSamples[self.man.folderDict['version']-1]
            self.query(opts, self.man.folderDict, self.man, NumSamples=NumSamples, NumSamplesMaxMCD=opts.NumSamplesMaxMCD)

            # idx_sort = self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, clustering=True, create_mask=False, create_pseudo=True, create_uc=True, pseudo_full=True, strata=['class'])

            # Annotate images
            self.annotation_auto(opts, self.man, self.man.datasets['action_round'].data)
            # Preprocessing raw data
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            # Train model
            self.train_step(opts, self.man.folderDict, pretrained=True)
            # Copy model and labeles
            self.copy_model_labels(opts, self.man.folderDict)
            self.man.update_status(self.man.folderDict, 'training', True)
            # Update round_status
            for key in self.man.folderDict['round_status']:
                self.man.update_status(self.man.folderDict, key, True)

    def reset(self, opts, copy_nnUNet_data=True, key_reset='selection_annotation_manual'):
        
        # self=alunet
        # copy_nnUNet_data=True
        # key_reset='selection_annotation_manual'
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _ = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        opts.alunet=self
        
        keys = [k for k in self.man.folderDict['round_status']]
        keys.reverse()  
        for k in keys:  
            if k=='training':
                #self.man.folderDict['round_status'][k]=False
                self.man.update_status(self.man.folderDict, k, False)
            if k=='selection_annotation_manual':
                NumSamples=opts.ALSamples[self.man.folderDict['version']-2]
                self.man.datasets['action_round'].data = self.man.datasets['train'].data[-NumSamples:]
                self.man.datasets['train'].data = self.man.datasets['train'].data[0:-NumSamples]
                # Reset labelsTr
                labelsTr = join(nnUNet_raw, opts.DName, 'labelsTr')
                shutil.rmtree(labelsTr)
                labelsTr_pre = join(self.man.folderDict['modelpath_prev'], 'labelsTr')
                shutil.copytree(labelsTr_pre, labelsTr)
                # Create action for segmentation
                #self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, create_mask=False, create_pseudo=True)
                # Reset status
                #self.man.folderDict['round_status'][k]=False
                self.man.update_status(self.man.folderDict, k, False)
                # Save
                self.man.save(include=['train', 'action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            if k==key_reset:
                break
    
    def extract_data_action_select(self, opts, fp_manual):
        if opts.dim==-5:
            print('Doing selection_subset')
            # Read action list
            fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
            actionlist = ALAction.load(fip_actionlist) 
            action_pre = self.man.datasets['action_pre'].data
            for a in actionlist:
                for s in action_pre:
                    if a.id==s.ID:
                        s.action = a
                        a.filetype='.nii.gz'
            action_pre = [s for s in action_pre if hasattr(s, "action")]

            # Check if samples are classified as positive and negative
            classes = [(s.F['imagename'], s.action.info['class']) for s in action_pre if 'class' in s.action.info]
            df_classes = pd.DataFrame()
            for cl in classes:
                imname = cl[0]
                for sl in cl[1]:
                    #df_classes=df_classes.append({'imagename': imname, 'slice': sl[0], 'class': sl[1]}, ignore_index=True)
                    df_classes = pd.concat([df_classes, pd.DataFrame([{'imagename': imname, 'slice': sl[0], 'class': sl[1]}])], ignore_index=True)

            
            data_query = self.man.datasets['query'].data
            data_action_select = []
            for index, row in df_classes.iterrows():
                for s in action_pre+data_query:
                    if (s.F['imagename']==row['imagename']) and (s.F['lbs_org'][0]==row['slice']):
                        s.action = ALAction()
                        s.action.info['class']=row['class']
                        data_action_select.append(s)
        else:
            data = self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data + self.man.datasets['query'].data
            file1 = open(os.path.join(fp_manual, 'XALabelerLUT.ctbl'), 'r')
            lines = file1.readlines()
            for line in lines:
                if line.split(' ')[1]=='positive':
                    label_positive=int(line.split(' ')[0])
                if line.split(' ')[1]=='negative':
                    label_negative=int(line.split(' ')[0])
                    
            imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
            data_action_select=[]
            for imagename in imagenames:
                fip_label = join(fp_manual, 'refine', imagename + '.nii.gz')
                if os.path.isfile(fip_label):
                    #sys.exit()
                    ref = CTRef(fip_label).ref()
                    label_pos = skimage.measure.label(ref==label_positive)
                    props_pos = skimage.measure.regionprops_table(label_pos, properties=['centroid', 'area', 'label'])
                    idx_pos = np.ones((props_pos['centroid-0'].shape[0]))*-1
                    dist_pos = np.ones((props_pos['centroid-0'].shape[0]))*1000000
                    label_neg = skimage.measure.label(ref==label_negative)
                    props_neg = skimage.measure.regionprops_table(label_neg, properties=['centroid', 'area', 'label'])                      
                    idx_neg = np.ones((props_neg['centroid-0'].shape[0]))*-1
                    dist_neg = np.ones((props_neg['centroid-0'].shape[0]))*1000000
                    for i,s in enumerate(data):
                        if s.F['imagename']==imagename:
                            #sys.exit()
                            bL=s.F['lbs_org']
                            bU=s.F['ubs_org']
                            x = np.mean([bL[0],bU[0]])
                            y = np.mean([bL[1],bU[1]])
                            z = np.mean([bL[2],bU[2]])
                            # Positive samples
                            distp = ((x-props_pos['centroid-0'])*(x-props_pos['centroid-0']))+((y-props_pos['centroid-1'])*(y-props_pos['centroid-1']))+((z-props_pos['centroid-2'])*(z-props_pos['centroid-2']))
                            idxp = np.where(distp<dist_pos)
                            dist_pos[idxp] = distp[idxp]
                            idx_pos[idxp] = i
                            # Negative samples
                            distn = ((x-props_neg['centroid-0'])*(x-props_neg['centroid-0']))+((y-props_neg['centroid-1'])*(y-props_neg['centroid-1']))+((z-props_neg['centroid-2'])*(z-props_neg['centroid-2']))
                            idxn = np.where(distn<dist_neg)
                            dist_neg[idxn] = distn[idxn]
                            idx_neg[idxn] = i
                    
                    idx_pos = np.unique(idx_pos)
                    idx_neg = np.unique(idx_neg)
                    for i in list(idx_pos):
                        s = data[int(i)]
                        s.action = ALAction()
                        s.action.info['class']=1
                        data_action_select.append(s)
                    for i in list(idx_neg):
                        s = data[int(i)]
                        s.action = ALAction()
                        s.action.info['class']=2
                        data_action_select.append(s)

        
    def alround_target(self, opts, copy_nnUNet_data=True):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        # copy_nnUNet_data=False

        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _ = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        #print('load_1234')
        
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False, load_dict=dict({'del_grad': True}))
        
        #fp_manual = self.man.folderDict['fp_manual']
        #NumSamples=opts.ALSamples[self.man.folderDict['version']-2]
        
        opts.alunet=self

        # Set number samples
        #NumSamples=opts.ALSamples[self.man.folderDict['version']-1]
        NumSamples=opts.ALSamples[self.man.folderDict['version']]
        print('NumSamples AL', NumSamples)
        NumSamplesCheck = NumSamples
        NumSamplesPre = max(500,NumSamples*10)
        
        # Create new training round
        processed = np.all([self.man.folderDict['round_status'][key] for key in self.man.folderDict['round_status']])
        if processed: 
            _, _ = self.alcreate(opts, copy_nnUNet_data=True)
        self.man.update_status(self.man.folderDict, 'correction_proposal', True)
        self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
        self.man.update_status(self.man.folderDict, 'correction_preparation', True)
        self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)

        # Selection proposal
        if not self.man.folderDict['round_status']['selection_proposal']:
            print('Doing selection_proposal')
            # Disable fastmode
            fastmode = opts.fastmode
            opts.fastmode = False
            # Query new samples
            strat = opts.strategy
            opts.strategy='UCPROP'
            self.query(opts, self.man.folderDict, self.man, NumSamples=NumSamplesPre, NumSamplesMaxMCD=opts.NumSamplesMaxMCD)
            opts.strategy = strat
            self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
            self.man.datasets['action_round'].data=[]
            self.man.save(include=['action_round', 'action_pre'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            _ = self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, clustering=False, create_mask=False, create_pseudo=True, create_uc=True, pseudo_full=True)

            # Update status
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)


        # Selection of targets (manual)
        if not self.man.folderDict['round_status']['selection_target_manual']:
            print('Doing selection_target_manual')
            sys.exit('Please classifie images in positive and negative using XALabeler software!')
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
        
        # Subset selection based on selected target samples
        if not self.man.folderDict['round_status']['selection_subset']:
            data = self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data + self.man.datasets['query'].data 
            fip_actionlist = os.path.join(self.man.folderDict['fp_manual'], 'actionlist.json')
            actionlist = ALAction.load(fip_actionlist) 
            action_pre = self.man.datasets['action_pre'].data
            data_select = []
            for a in actionlist:
                for s in action_pre:
                    if a.id==s.ID:
                        s.action = a
                        a.filetype='.nii.gz'
                        if a.info['label']['batch']:
                            data_select.append(s)
            print('data_select123', len(data_select))
            self.man.datasets['query'].delete(data_select)
            self.man.datasets['action_round'].data = data_select
            self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            self.man.update_status(self.man.folderDict, 'selection_subset', True)

        # Annotated images (manual)
        if not self.man.folderDict['round_status']['selection_annotation_manual']:
            print('Doing selection_annotation_manual')
            if opts.segauto:
                self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
            else:
                if not opts.interactive:
                    sys.exit('Please annotate images using XALabeler software!')
                    self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                else:
                    self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                
        if not self.man.folderDict['round_status']['training']:
            print('Doing training')
            # Annotate 
            if opts.segauto:
                self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
            else:
                data = self.man.datasets['action_round'].data + self.man.datasets['query'].data + self.man.datasets['train'].data
                _ = self.annotation_manual(opts, self.man, data, bg_offset=False)

            # Preprocess data
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier + " -npfp 1 -np 1 " + "--verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            # Train model
            self.train_step(opts, self.man.folderDict, pretrained=True)
            # Copy model and labeles
            self.copy_model_labels(opts, self.man.folderDict)
            self.man.update_status(self.man.folderDict, 'training', True)
            

    
    def alfull(self, opts):
        
        # self=alunet
        # Create new AL round  
        NewVersion = True
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True)
        man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Update training set
        man.datasets['train'].data = man.datasets['query'].data
        man.save(include=['train, query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        
        # Update annotation
        if opts.label_manual:
            pass
            #self.annotation_manual(man.datasets['train'].data)
        else:
            self.annotation_auto(opts, man, data=man.datasets['train'].data, full=True)

        # Preprocessing raw data
        #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
        subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -pl " + opts.plans_identifier + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
        
        # Train model
        self.train(opts)
        
        # Copy model and labeles
        self.copy_model_labels(opts, folderDict)
        
# ---------------------------------------------------------------------------
# Active-learning persistence and nnU-Net integration
# ---------------------------------------------------------------------------


class UNetManagerV2(ALManager):
    """Manage active-learning datasets, round folders, and nnU-Net models."""

    def __init__(self, fp_dataset):
        ALManager.__init__(self, fp_dataset)
        self.fp_dataset = fp_dataset
        for key in self.datasets.keys():
            self.datasets[key] = UNetDatasetV2(key)
        self.datasets['action']=UNetDatasetV2('action')
        self.datasets['action_round']=UNetDatasetV2('action_round')
        self.datasets['action_pre']=UNetDatasetV2('action_pre')
        
    def createALFolderpath(self, opts, fip_split=None, method=None, NewVersion=False, VersionUse=None, copy_prev_labelsTr=False, copy_nnUNet_data=True):
        print('createALFolderpath123')
        folderDict = super().createALFolderpath(fp_active=opts.fp_active, fip_split=fip_split, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=method, NewVersion=NewVersion, VersionUse=VersionUse)
        #folderDict['fip_hdf5_all'] = os.path.join(opts.fp_active, 'hdf5_all_'+str(opts.dim)+'D.hdf5')
        # Copy nnunet dataset
        if folderDict['copy_init']:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
            fp_nnUNet_results = os.path.join(fp_nnunet, 'nnUNet_results')
            #dnameInit = 'Dataset' + '1'+opts.dataset_name_or_id[1:2]+'0' + '_' + opts.dataset
            #dnameInit = 'Dataset' + opts.dataset_name_or_id[0]+'00' + '_' + opts.dataset
            #dnameInit = 'Dataset' + opts.dataset_name_or_id[0:2]+'0' + '_' + opts.dataset
            dnameInit = 'Dataset' + opts.dataset_name_or_id_init + '_' + opts.dataset
            opts.dataset_name_or_id
            if copy_nnUNet_data:
                print('Copy nnUNet_raw')
                print('fp_labelsTr1234', os.path.join(fp_nnUNet_raw, dname))
                #sys.exit()
                copy_tree(os.path.join(fp_nnUNet_raw, dnameInit), os.path.join(fp_nnUNet_raw, dname))
                print('Copy nnUNet_preprocessed')
                copy_tree(os.path.join(fp_nnUNet_preprocessed, dnameInit), os.path.join(fp_nnUNet_preprocessed, dname))
                print('Copy nnUNet_result')
                copy_tree(os.path.join(fp_nnUNet_results, dnameInit), os.path.join(fp_nnUNet_results, dname))
            
            # Change dataset_name in nnUUnetPlans.json
            plans_file = os.path.join(nnUNet_preprocessed, dname, opts.plans_identifier+'.json')
            plans = load_json(plans_file)
            plans['dataset_name'] = dname
            save_json(plans, plans_file)
            
        # Reset labelsTr from prvious round
        
        #if folderDict['version']>2 and copy_prev_labelsTr:
        #if folderDict['version']>2 and folderDict['newVersion'] and copy_prev_labelsTr:
        if folderDict['version']>1 and folderDict['newVersion'] and copy_prev_labelsTr:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
            fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
            
            fp_labelsTr_prev = os.path.join(folderDict['modelpath_prev'], 'labelsTr')
            if os.path.exists(fp_labelsTr):
                shutil.rmtree(fp_labelsTr)
            shutil.copytree(fp_labelsTr_prev, fp_labelsTr)
            
        # Store round infomation
        folderDict['round_status']=OrderedDict({'correction_proposal': False,
                                                'correction_selection_manual': False, 
                                                'correction_preparation': False, 
                                                'correction_annotation_manual': False,
                                                'selection_proposal': False, 
                                                'selection_target_manual': False, 
                                                'selection_subset': False, 
                                                'selection_annotation_manual': False,
                                                'training': False})
        folderDict['fip_round_status'] = join(folderDict['fp_manager'], 'round_status.json')
        if os.path.isfile(folderDict['fip_round_status']) and not NewVersion:
            with open(folderDict['fip_round_status'], 'r') as f:
                folderDict['round_status'] = json.load(f)
        else:
            with open(folderDict['fip_round_status'], 'w') as json_file:
                json.dump(folderDict['round_status'], json_file, indent = 4)
            
        if opts.fp_manual is None:
            folderDict['fp_manual'] = join(os.path.dirname(self.folderDict['modelpath']), 'data_manual')
        else:
            folderDict['fp_manual'] = opts.fp_manual
        # Update path for nnUNet
        #ALFolderpathMethod = os.path.join(fp_active, method)
        #if not os.path.isdir(ALFolderpathMethod) and not method=='INIT':
        #folderDict['modelpath'] = 
        self.folderDict=folderDict
        return folderDict
    
    def update_status(self, folderDict, key, value):
        if key not in folderDict['round_status']:
            raise ValueError("Key " + key + " not found in status dict.")
        folderDict['round_status'][key]=value
        with open(folderDict['fip_round_status'], 'w') as json_file:
            json.dump(folderDict['round_status'], json_file, indent = 4)
    
    def init_patches(self, opts):
        
        # self=self.man
        print('init_patches123', join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration))
        
        tile_step_size=0.5
        #images = glob(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration + '/*.npz'))
        images = glob(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration + '/*.b2nd'))
        images = [x for x in images if 'seg' not in x]

        patch_size = load_json(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json'))['configurations'][opts.configuration]['patch_size']
        if len(patch_size)==2: patch_size=[1]+patch_size
        patch_size = np.array(patch_size).astype(np.int32)
        plans_file = os.path.join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json')
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        spacing_after_resampling = plans['configurations'][opts.configuration]['spacing']
        if len(spacing_after_resampling)==2: spacing_after_resampling=[1.0]+spacing_after_resampling
        dim = opts.dim
        ID=0
        data=[]
        for image in tqdm(images):
            #imname = os.path.basename(image).split('.')[0]
            imname = splitFilePath(image)[1]
            #if 'BUD-008' in imname:
            #     sys.exit()
            #sys.exit()
            pkl = pd.read_pickle(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration, imname+'.pkl'))
            #fip = os.path.join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_' + opts.configuration, image)
            #shape_after_resampling = np.load(fip)['seg'].shape[1:]

            fip_seg = os.path.join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_' + opts.configuration, imname + '_seg.b2nd')
            #print('fip_seg123', fip_seg)
            shape_after_resampling = np.array(blosc2.open(fip_seg)).shape[1:]

            # steps = compute_steps_for_sliding_window(pkl['shape_before_cropping'], patch_size, tile_step_size)
            steps = compute_steps_for_sliding_window(shape_after_resampling, patch_size, tile_step_size)
            # Steps have to be adjusted from 1 to 2 because the sliding window with 0.5 overlab would create multiple images from same slice
            if opts.dim==2:
                steps[0] = [i for i in range(shape_after_resampling[0])]
            
            #if len(steps)==2: steps=steps+[[None]]
            for sx in list(steps[0]):
                for sy in list(steps[1]):
                    for sz in list(steps[2]):
                        #sys.exit()
                        
                        if opts.dim==2:
                            need_to_pad = [patch_size[i+1] - shape_after_resampling[i+1] for i in range(dim)]
                            lbs = [- need_to_pad[i] // 2 for i in range(dim)]
                            #ubs = [shape_after_resampling[i+1] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i+1] for i in range(dim)]
                            coord = np.array([[0,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2),int(sz+patch_size[2]/2)]])
                            selected_voxel = coord[0][2:]
                            bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i+1] // 2) for i in range(dim)]
                            bbox_ubs = [bbox_lbs[i] + patch_size[i+1] for i in range(dim)]
                            #valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            #valid_bbox_ubs = [min(shape_after_resampling[i+1], bbox_ubs[i]) for i in range(dim)]
                            padding = [(0,0)]+[(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape_after_resampling[i+1], 0)) for i in range(dim)]
                          
                            patch = UNetPatchV2()
    
                            patch.ID=ID
                            patch.F['props']=dict()
                            patch.F['ID']=ID
                            #patch.F['imagename']=os.path.basename(image).split('.')[0]
                            patch.F['imagename']=splitFilePath(image)[1]
                            patch.F['classLocations']=coord
                            patch.F['spacing']=np.array(pkl['spacing'])
                            patch.F['shape_before_cropping']=pkl['shape_before_cropping']
                            patch.F['bbox_used_for_cropping']=pkl['bbox_used_for_cropping']
                            patch.F['shape_after_cropping_and_before_resampling']=pkl['shape_after_cropping_and_before_resampling']
                            patch.F['shape_after_resampling']=shape_after_resampling
                            patch.F['patch_size']=patch_size
                            patch.F['spacing_after_resampling']=spacing_after_resampling
                            patch.F['padding']=padding
                            # Defined properties
                            ratio=patch.F['spacing_after_resampling']/patch.F['spacing']
                            if opts.dim==2:
                                ratio[0]=1.0
                            patch.F['lbs_res'] = np.array([sx+padding[0][0], sy+padding[1][0], sz+padding[2][0]]).astype(np.int32)
                            patch.F['ubs_res'] = np.array([sx+patch_size[0]-padding[0][1], sy+patch_size[1]-padding[1][1], sz+patch_size[2]-padding[2][1]]).astype(np.int32)
                            patch.F['lbs_crop'] = (np.round(patch.F['lbs_res']*ratio)).astype(np.int32)
                            patch.F['ubs_crop'] = (np.round(patch.F['ubs_res']*ratio)).astype(np.int32)
                            patch.F['lbs_org'] = np.array([patch.F['lbs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim+1)])
                            patch.F['ubs_org'] = np.array([patch.F['ubs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim+1)])
    
                            # Update minimum and maximum lbs_crop and ubs_crop
                            patch.F['lbs_org'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))
                        
                        else:
                            need_to_pad = [patch_size[i] - shape_after_resampling[i] for i in range(dim)]
                            lbs = [- need_to_pad[i] // 2 for i in range(dim)]
                            #ubs = [shape_after_resampling[i+1] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i+1] for i in range(dim)]
                            coord = np.array([[0,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2),int(sz+patch_size[2]/2)]])
                            selected_voxel = coord[0][1:]
                            bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i] // 2) for i in range(dim)]
                            bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]
                            #valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            #valid_bbox_ubs = [min(shape_after_resampling[i+1], bbox_ubs[i]) for i in range(dim)]
                            padding = [(0,0)]+[(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape_after_resampling[i], 0)) for i in range(dim)]
                
                            patch = UNetPatchV2()
                            patch.ID=ID
                            patch.F['props']=dict()
                            patch.F['ID']=ID
                            #patch.F['imagename']=os.path.basename(image).split('.')[0]
                            patch.F['imagename']=splitFilePath(image)[1]
                            patch.F['classLocations']=coord
                            patch.F['spacing']=np.array(pkl['spacing'])
                            patch.F['shape_before_cropping']=pkl['shape_before_cropping']
                            patch.F['bbox_used_for_cropping']=pkl['bbox_used_for_cropping']
                            patch.F['shape_after_cropping_and_before_resampling']=pkl['shape_after_cropping_and_before_resampling']
                            patch.F['shape_after_resampling']=shape_after_resampling
                            patch.F['patch_size']=patch_size
                            patch.F['spacing_after_resampling']=spacing_after_resampling
                            patch.F['padding']=padding
                            # # Defined properties
                            # ratio=patch.F['spacing_after_resampling']/patch.F['spacing']
                            # if opts.dim==2:
                            #     ratio[0]=1.0
                            # patch.F['lbs_res'] = np.array([sx+padding[0][0], sy+padding[1][0], sz+padding[2][0]]).astype(np.int32)
                            # patch.F['ubs_res'] = np.array([sx+patch_size[0]-padding[0][1], sy+patch_size[1]-padding[1][1], sz+patch_size[2]-padding[2][1]]).astype(np.int32)
                            # patch.F['lbs_crop'] = (np.round(patch.F['lbs_res']*ratio)).astype(np.int32)
                            # patch.F['ubs_crop'] = (np.round(patch.F['ubs_res']*ratio)).astype(np.int32)
                            # patch.F['lbs_org'] = np.array([patch.F['lbs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
                            # patch.F['ubs_org'] = np.array([patch.F['ubs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
    
                            # # Update minimum and maximum lbs_crop and ubs_crop
                            # patch.F['lbs_org'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            # patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))

                                                        # Defined properties
                            ratio=patch.F['spacing_after_resampling']/patch.F['spacing']
                            if opts.dim==2:
                                ratio[0]=1.0
                            #patch.F['lbs_res'] = np.array([sx+padding[0][0], sy+padding[1][0], sz+padding[2][0]]).astype(np.int32)
                            patch.F['lbs_res'] = np.array([sx+padding[1][0], sy+padding[2][0], sz+padding[3][0]]).astype(np.int32)
                            patch.F['ubs_res'] = np.array([sx+patch_size[0]-padding[0][1], sy+patch_size[1]-padding[1][1], sz+patch_size[2]-padding[2][1]]).astype(np.int32)
                            patch.F['lbs_crop'] = (np.round(patch.F['lbs_res']*ratio)).astype(np.int32)
                            patch.F['ubs_crop'] = (np.round(patch.F['ubs_res']*ratio)).astype(np.int32)
                            patch.F['lbs_org'] = np.array([patch.F['lbs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
                            patch.F['ubs_org'] = np.array([patch.F['ubs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
                            patch.F['s'] = (sx, sy, sz)
    
                            # Update minimum and maximum lbs_crop and ubs_crop
                            #patch.F['lbs_org'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            #patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))

                            #patch.F['lbs_crop'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            #patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))

                            #if patch.F['imagename']=='hippocampus_245':
                            #    print('hippocampus_245', patch.F)
                            #    sys.exit()
                        
                        data.append(patch)   
                        ID=ID+1

        dataset_name_or_id = opts.dataset_name_or_id
        configuration = opts.configuration
        tr=opts.nnUNetTrainer
        p=opts.plans_identifier
        device = torch.device('cuda')
        from tools.nnUNet.nnUNet.nnunetv2.run.run_training import get_trainer_from_args
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, device)

        #nnunet_trainer = get_trainer_from_args(opts.dataset_name_or_id, opts.configuration, opts.fold, opts.nnUNetTrainer, opts.plans_identifier, False)
        split = nnunet_trainer.do_split()
        
        # Copy split file
        fip_split_nnunet = os.path.join(nnUNet_preprocessed, opts.DName, 'splits_final.json')
        shutil.copyfile(fip_split_nnunet, opts.fip_split)
        
        # Split validation data
        data_split = json.load(open(opts.fip_split))
        images_valid = data_split[opts.fold]['val']
        data_valid=[]
        data_query=[]
        for s in data:
            #if s.F['imagename'].split('.')[0] in images_valid:
            if s.F['imagename'] in images_valid:
                data_valid.append(s)
            else:
                data_query.append(s)
        self.datasets['valid'].data = data_valid
        self.datasets['query'].data = data_query
        
        
    def load_model(self, opts, folderDict, previous=False, const_dropout=False, load_weights=True):
        # self=man

        dataset_name_or_id = opts.dataset_name_or_id
        configuration = opts.configuration
        tr=opts.nnUNetTrainer
        p=opts.plans_identifier
        #use_compressed='False'
        device = torch.device('cuda')
        from tools.nnUNet.nnUNet.nnunetv2.run.run_training import get_trainer_from_args
        #nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, use_compressed)
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, device)
        
        #nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, use_compressed)
        settingsfilepath_model = os.path.join(folderDict['modelpath'], 'LITS.yml')
        net = ALUNETMODEL2(settingsfilepath_model, overwrite=True)
        net.model = {'unet': nnunet_trainer}
        if previous:
            net.props['fip_checkpoint']=folderDict['modelpath_prev']
        else:
            net.props['fip_checkpoint']=folderDict['modelpath']
        if load_weights:
            #net.model['unet'].load_checkpoint(os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_'+str(opts.fold), 'checkpoint_best.pth'))
            net.model['unet'].load_checkpoint(os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_'+str(opts.fold), 'checkpoint_final.pth'))
            print('Loading model:', os.path.join(net.props['fip_checkpoint']))
        else:
            net.model['unet'].initialize()
        return net      
      
    
class UNetDatasetV2(SALDataset):
    """Dataset container used by :class:`UNetManagerV2`."""

    def __init__(self, name=''):
        SALDataset.__init__(self, name)
            
class UNetSampleV2(ALSample):
    """Active-learning sample with image, mask, and prediction fields."""

    def __init__(self):
        ALSample.__init__(self)
        self.Xlabel = ['XImage']
        self.Ylabel = ['XMask']
        self.Plabel = ['XMaskPred']
        self.patches = []
               
        
    @classmethod
    def sort_F(cls, sl, prop='imagename', prop_sorted=False):
        if prop_sorted:
            props = sorted(np.unique([s.F[prop] for s in sl]))
        else:
            props = [s.F[prop] for s in sl]
            indexes = np.unique(props, return_index=True)[1]
            props = [props[index] for index in sorted(indexes)]
        slout=[]
        idx_sort=[]
        for pr in props:
            for i,s in enumerate(sl):
                if s.F['imagename']==pr:
                    slout.append(s)
                    idx_sort.append(i)
        return slout, idx_sort

    @classmethod
    def savePseudo(cls, opts, data, folderDict, fp_pseudo, filetype, pseudo_full=False, tile_step_size=0.5):
        # cls=ALSegmentCACSSample
        # data=data_samples
        
        dataSort,_ = UNetPatchV2.sort_F(data, prop='imagename')
        
        plans = load_json(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json'))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(opts.configuration)
        
        #idx = np.argsort([s.F['imagename'] for s in dataSort])
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        #imagenames = list(np.sort([s.F['imagename'] for s in dataSort]))
        imagenames = list([s.F['imagename'] for s in dataSort])
        
        print('pseudo_full12378', pseudo_full)
        if pseudo_full:
            imagenames = list(np.unique(imagenames))
            for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
                print('fp_imagesTr123', fp_imagesTr)
                print('imn123', imn)
                fip_image = glob(fp_imagesTr + '/'+imn+'_*')[0]
                print('fip_image123', fip_image)
                fip_pseudo = os.path.join(fp_pseudo, imn + filetype)
                modelpath = opts.alunet.man.folderDict['modelpath_prev']
                opts.alunet.predict_image(opts, folderDict, modelpath, fip_image, fip_pseudo, tile_step_size=tile_step_size)
        
        else:
            for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
                s = dataSort[i]
                if i==0 or imagenames[i]!=imagenames[i-1]:
                    #fip_pseudo = glob(join(fp_labelsTr, imn)+'.*')[0]
                    fip_pseudo = join(fp_labelsTr, imn+filetype)
                    ref = CTRef(fip_pseudo)
                    arr = ref.ref()
                    arr = arr*0
    
                original_spacing = s.F['spacing_after_resampling']
                target_spacing = s.F['spacing']
                padding = s.F['padding']
                # Reshape pseudo label
                XMaskPred = s.P['XMaskPred']
                # Invert padding
                XMaskPredPad = XMaskPred[:,padding[0][0]:XMaskPred.shape[1]-padding[0][1],padding[1][0]:XMaskPred.shape[2]-padding[1][1], padding[2][0]:XMaskPred.shape[3]-padding[2][1]]
                # Invert rescale
                # !!!!
                if s.F['ubs_org'][1]>512:
                    s.F['ubs_org'][1]=512
                if s.F['ubs_org'][2]>512:
                    s.F['ubs_org'][2]=512
                shape_tar_seg = (XMaskPredPad.shape[1], s.F['ubs_org'][1]-s.F['lbs_org'][1], s.F['ubs_org'][2]-s.F['lbs_org'][2])
                XMaskPredProp = torch.from_numpy(configuration_manager.resampling_fn_data(XMaskPredPad, shape_tar_seg, original_spacing, target_spacing))
                XMaskPredHot = torch.argmax(compute_one_hot_torch(XMaskPredProp), dim=1)
                
                print('ups123', s.F['ubs_org'])
                print('lps123', s.F['lbs_org'])
                print('shape_tar_seg123', shape_tar_seg)
                print('XMaskPredHot123', XMaskPredHot.shape)
                arr[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]=XMaskPredHot
                
                if i==len(imagenames)-1 or imn!=imagenames[i+1]:
                    ref_out = CTRef(arr)
                    ref_out.copyInformationFrom(ref)
                    name_pseudo = imn + filetype
                    fip_pseudo = os.path.join(fp_pseudo, name_pseudo)
                    ref_out.save(fip_pseudo)

        
class UNetPatchV2(UNetSampleV2):
    """Patch-level sample with serialization and visualization helpers."""

    def __init__(self):
        UNetSampleV2.__init__(self)
        #self.IDP=None
        self.F['props']=None
        
    @classmethod
    def save(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        if len(sl)>0:
            if hdf5:
                cls.save_hdf5(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)
            else:
                cls.save_pkl(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)

    @classmethod
    def save_pkl(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        os.makedirs(fp_data, exist_ok=True)
        df = pd.DataFrame()
        fip_df = os.path.join(fp_data, 'sl.pkl')
        features = [s.F.copy() for s in tqdm(sl)]
        df = pd.DataFrame.from_records(features)
        df.to_pickle(fip_df)
        

    @classmethod
    def load(cls, fp_data, load_dict, load_class, dataset_name='', hdf5=False):
        if hdf5:
            return cls.load_hdf5(fp_data=fp_data, load_dict=load_dict, load_class=load_class, dataset_name=dataset_name)
        else:
            return cls.load_pkl(fp_data=fp_data, load_dict=load_dict, dataset_name=dataset_name)

    @classmethod
    def load_pkl(cls, fp_data, load_dict, dataset_name=''):
        data=[]
        fip_df = os.path.join(fp_data, 'sl.pkl')
        if os.path.isfile(fip_df):
            df = pd.read_pickle(fip_df)
            for index, row in df.iterrows():
                s = cls()
                s.ID = row['ID']
                s.name = str(row['ID'])
                s.F=dict(row)
                if 'del_grad' in load_dict:
                    if load_dict['del_grad']:
                        #print('del_grad123')
                        if 'grad' in s.F: del s.F['grad']
                        if 'FI' in s.F: del s.F['FI']
                        if 'uc' in s.F: del s.F['uc']
                #print('del_grad123', 'del_grad' in load_dict)
                #sys.exit('OUT05')
                data.append(s)
            #print('del_grad123', 'del_grad' in load_dict)
            #sys.exit('OUT06')
        return data

    @classmethod
    def saveUC(cls, opts, data, folderDict, fp_uc, filetype):
        # cls=ALSegmentCACSSample
        # data=data_samples
        
        dataSort,_ = UNetPatchV2.sort_F(data, prop='imagename')
        
        plans = load_json(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json'))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(opts.configuration)
        
        #idx = np.argsort([s.F['imagename'] for s in dataSort])
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        #imagenames = list(np.sort([s.F['imagename'] for s in dataSort]))
        imagenames = list([s.F['imagename'] for s in dataSort])
    
        for i,imn in enumerate(tqdm(imagenames, desc='Save uc map')):
            s = dataSort[i]
            if i==0 or imagenames[i]!=imagenames[i-1]:
                #fip_pseudo = glob(join(fp_labelsTr, imn)+'.*')[0]
                fip_pseudo = join(fp_labelsTr, imn+filetype)
                im = CTImage(fip_pseudo)
                arr = im.image()
                arr = arr*0.0

            original_spacing = s.F['spacing_after_resampling']
            target_spacing = s.F['spacing']
            padding = s.F['padding']
            # Reshape pseudo label
            #XMaskPred = s.P['XMaskPred']
            Xuc = s.F['ucMap']
            Xuc = Xuc.unsqueeze(0)
            # Invert padding
            if opts.dim==2:
                XucPad = Xuc[:,padding[0][0]:Xuc.shape[1]-padding[0][1],padding[1][0]:Xuc.shape[2]-padding[1][1], padding[2][0]:Xuc.shape[3]-padding[2][1]]
            else:
                print('padding123', padding)
                print('Xuc123', Xuc.shape)
                #XucPad = Xuc[0,:,padding[0][0]:Xuc.shape[1]-padding[0][1],padding[1][0]:Xuc.shape[2]-padding[1][1], padding[2][0]:Xuc.shape[3]-padding[2][1], padding[3][0]:Xuc.shape[4]-padding[3][1]]
                XucPad = Xuc[0,:,padding[1][0]:Xuc.shape[2]-padding[1][1], padding[2][0]:Xuc.shape[3]-padding[2][1], padding[3][0]:Xuc.shape[4]-padding[3][1]]
            # Invert rescale
            # !!!! Check if necessary
            #if s.F['ubs_org'][1]>512:
            #    s.F['ubs_org'][1]=512
            #if s.F['ubs_org'][2]>512:
            #    s.F['ubs_org'][2]=512
            

            if opts.dim==2:
                shape_tar_seg = (XucPad.shape[1], s.F['ubs_org'][1]-s.F['lbs_org'][1], s.F['ubs_org'][2]-s.F['lbs_org'][2])
                # !!! -  Target shape correction - Why necessary? Rounding error?
                if shape_tar_seg[1]>arr.shape[1]:
                    shape_tar_seg = (shape_tar_seg[0], arr.shape[1], shape_tar_seg[2])
                if shape_tar_seg[2]>arr.shape[2]:
                    shape_tar_seg = (shape_tar_seg[0], shape_tar_seg[1], arr.shape[2])
            else:
                print('s.Fubs_org23', s.F['ubs_org'])
                print('s.Flbs_org23', s.F['lbs_org'])
                print('XucPad567', XucPad.shape)
                print('arr.shape', arr.shape)
                print('fip_pseudo123', fip_pseudo)
                print('Fimganemae', s.F['imagename'])
                # !!! Check why s.Fubs_org23 is in some cases larger than image size. Rounding error?
                if s.F['ubs_org'][0]>arr.shape[0]:
                    s.F['ubs_org'] = (arr.shape[0], s.F['ubs_org'][1], s.F['ubs_org'][2])
                if s.F['ubs_org'][1]>arr.shape[1]:
                    s.F['ubs_org'] = (s.F['ubs_org'][0], arr.shape[1], s.F['ubs_org'][2])
                if s.F['ubs_org'][2]>arr.shape[2]:
                    s.F['ubs_org'] = (s.F['ubs_org'][0], s.F['ubs_org'][1], arr.shape[2])
                shape_tar_seg = (s.F['ubs_org'][0]-s.F['lbs_org'][0], s.F['ubs_org'][1]-s.F['lbs_org'][1], s.F['ubs_org'][2]-s.F['lbs_org'][2])
                print('shape_tar_seg123', shape_tar_seg)
                # !!! -  Target shape correction - Why necessary? Rounding error?
                if shape_tar_seg[0]>arr.shape[0]:
                    shape_tar_seg = (arr.shape[0], shape_tar_seg[1], shape_tar_seg[2])
                if shape_tar_seg[1]>arr.shape[1]:
                    shape_tar_seg = (shape_tar_seg[0], arr.shape[1], shape_tar_seg[2])
                if shape_tar_seg[2]>arr.shape[2]:
                    shape_tar_seg = (shape_tar_seg[0], shape_tar_seg[1], arr.shape[2])

                
                # if shape_tar_seg[0]!=arr.shape[0]:
                #     shape_tar_seg = (arr.shape[0], shape_tar_seg[1], shape_tar_seg[2])
                # if shape_tar_seg[1]!=arr.shape[1]:
                #     shape_tar_seg = (shape_tar_seg[0], arr.shape[1], shape_tar_seg[2])
                # if shape_tar_seg[2]!=arr.shape[2]:
                #     shape_tar_seg = (shape_tar_seg[0], shape_tar_seg[1], arr.shape[2])

            # # !!! -  Target shape correction - Why necessary? Rounding error?
            # if shape_tar_seg[1]>arr.shape[1]:
            #     shape_tar_seg = (shape_tar_seg[0], arr.shape[1], shape_tar_seg[2])
            # if shape_tar_seg[2]>arr.shape[2]:
            #     shape_tar_seg = (shape_tar_seg[0], shape_tar_seg[1], arr.shape[2])
            #     print('imagename', s.F['imagename'])
            #     print('lbs_org', s.F['lbs_org'])



            XMaskPredProp = torch.from_numpy(configuration_manager.resampling_fn_data(XucPad, shape_tar_seg, original_spacing, target_spacing))
            print('XMaskPredProp123', XMaskPredProp.shape)

            if opts.dim==2:
                XMaskPredProp = XMaskPredProp.sum(dim=1)
            else:
                XMaskPredProp = XMaskPredProp.sum(dim=0)
            #XMaskPredHot = torch.argmax(compute_one_hot_torch(XMaskPredProp), dim=1)
            
            #arr[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]=XMaskPredHot
            print('arr123', arr.shape)
            arr[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]=XMaskPredProp

            if i==len(imagenames)-1 or imn!=imagenames[i+1]:

                # Normalize arr
                arr = arr/np.max(arr)*1000
                im_out = CTImage(arr)
                im_out.copyInformationFrom(im)
                name_uc = imn + filetype
                fip_pseudo = os.path.join(fp_uc, name_uc)
                #im_out.image_sitk = sitk.DICOMOrient(im_out.image_sitk, 'LPS')
                im_out.save(fip_pseudo)


    def __plot_image(self, image, name, title, color=False, save=False, filepath=None, format_im=None, dpi=300):
        if color:
            #print('image1234', image.shape)
            im = np.zeros((image.shape[2], image.shape[3]))
            for c in range(image.shape[1]):
                im = im + (c+1) * image[0,c,:,:]
            plt.imshow(im)
            plt.imshow(im, cmap='Accent', interpolation='nearest')
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.show()
        else:
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.imshow(image[0,0,:,:], cmap='gray')
            plt.show()
            
    def plotSample(self, plotlist=['XImage', 'XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine'], save=False, fp='', name='', color=False, title=True, format_im='svg', dpi=300):
        self.cpu()
        filepath = os.path.join(fp, name)

        for d in self.da:
            if d in plotlist:
                plotlist = plotlist + list(getattr(self, d, None).keys())

        for pl in plotlist:
            name = self.name + '_' + pl  + '_' + str(int(self.ID))
            im = self.getXY(pl)
            if im is not None:
                image = im.data.numpy()
                print('image.shape123', image.shape)
                if pl=='XImage' and len(image.shape)==4:
                    idx = int((image.shape[1]-1)/2)
                    image = image[:,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, format_im=format_im, save=save)
                else:
                    #image = image[:,1:2]
                    self.__plot_image(image, name, title, color=color, filepath=filepath, format_im=format_im, save=save)
    
    def showImageJ(self, plot='XImage'):
        if plot=='XImage':
            arr=self.X['XImage'][0,0,:,:,:].detach().cpu().numpy()
        elif plot=='XMaskPred':
            arr=np.zeros((self.P['XMaskPred'].shape[2], self.P['XMaskPred'].shape[3], self.P['XMaskPred'].shape[4]))
            for i in range(self.P['XMaskPred'].shape[1]):
                arr=arr+self.P['XMaskPred'][0,i,:,:,:].detach().cpu().numpy()*i
        else:
            print('Plot method ' + plot + ' not defined!')
            return
        im = CTImage(arr)
        im.showImageJ()
        
                    
class ALUNETMODEL2(DLBaseModel):
    """Thin :class:`DLBaseModel` wrapper holding the nnU-Net trainer."""

    def __init__(self, settingsfilepath, overwrite=False):
        props = defaultdict(lambda: None,
            NumChannelsIn = 1,
            NumChannelsOut = 2,
            Input_size = (512, 512, 1),
            Output_size = (512, 512, 2),
            device = 'cuda',
            modelname = 'UNetSeg',
            savePretrainedEpochMin=0
        )
        DLBaseModel.__init__(self, settingsfilepath=settingsfilepath, overwrite=overwrite, props=props)
