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
from ct.ct import CTImage

class RANDOM(ALStrategy):
    def __init__(self, name='RANDOM'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, manager, data_query, CLSample=None, NumSamples=10, batchsize=None, pred_class=['XMask'], save_uc=False, previous=None, NumSamplesMaxMCD=5000):
        idx = [x for x in range(len(data_query))]
        random.shuffle(idx)
        idx = idx[0:NumSamples]
        samples = [data_query[i] for i in idx]
        return samples
