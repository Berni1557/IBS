import sys
import warnings
import torch
import numpy as np
from typing import Tuple, Union, List
import shutil
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from threadpoolctl import threadpool_limits
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from threadpoolctl import threadpool_limits

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.ALUNET.nnUNetTrainer_ALUNET import nnUNetTrainer_ALUNET
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm, get_matching_dropout, convert_conv_op_to_dim

from typing import Type
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
import time
import random

def get_matching_dropout(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_DropoutNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.Dropout
    elif dimension == 2:
        return nn.Dropout2d
    elif dimension == 3:
        return nn.Dropout3d
    
class nnUNetTrainer_ORGAN(nnUNetTrainer_ALUNET):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
        #self.num_epochs = 30
        #self.num_epochs = 3
        #self.num_iterations_per_epoch = 250
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.select_only_labeled_train = True
        self.select_only_labeled_valid = True
        self.save_every = 2
        self.single_sample_processing = True

    def initialize(self):
        print('was_initialized123', self.was_initialized)
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        from pydoc import locate
        conv_op = locate(arch_init_kwargs['conv_op'])
        dropout = get_matching_dropout(conv_op)
        arch_init_kwargs['dropout_op'] = dropout.__module__ + '.' + dropout.__name__
        arch_init_kwargs['dropout_op_kwargs'] = {"p":0.1}
        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

    def on_train_start(self):
        print('on_train_start12378')
        if not self.was_initialized:
            self.initialize()

        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        print('dataloader_train123')
        #print('self.dataloader_train_indices123', self.dataloader_train.data_loader.indices)
        #sys.exit()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)

        if self.is_ddp:
            dist.barrier()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")


    def run_training(self):

        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                #print('batch_id123', batch_id)
                
                #start_time = time.time()
                train_outputs.append(self.train_step(next(self.dataloader_train)))
                #print("Time03", (time.time() - start_time))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                   #print('batch_id_val123', batch_id)
                   val_outputs.append(self.validation_step(next(self.dataloader_val)))
                #print('val_outputs123', val_outputs)
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        #print('keys', batch.keys())
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        #for t in target:    
        #    print('target123', t.shape)
        #print('target1234', torch.unique(target[0], return_counts=True))
        #print('target1234')
        #print('train_step123')

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def get_dataloaders(self, single=False):

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        print('nnUNetDataLoader3DSPARSE123')

        dl_tr = nnUNetDataLoader3DSPARSE(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        
        #print('nnUNetDataLoader3DSPARSE135')
        #sys.exit()

        dl_val = nnUNetDataLoader3DSPARSE(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  #oversample_foreground_percent=self.oversample_foreground_percent,
                                  oversample_foreground_percent=0.0,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        #dl_val = nnUNetDataLoader3DSPARSE(dataset_val, self.batch_size,
        #                          self.configuration_manager.patch_size,
        #                          self.configuration_manager.patch_size,
        #                          self.label_manager,
        #                          oversample_foreground_percent=self.oversample_foreground_percent,
        #                          sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
        #                          probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0 or single==True:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        
        #print('SingleThreadedAugmenter123')

        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        #print('mt_gen_train123')

        return mt_gen_train, mt_gen_val


# def get_dataloaders(self):
#         if self.dataset_class is None:
#             self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

#         # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
#         # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
#         patch_size = self.configuration_manager.patch_size

#         # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
#         # outputs?
#         deep_supervision_scales = self._get_deep_supervision_scales()
#         (
#             rotation_for_DA,
#             do_dummy_2d_data_aug,
#             initial_patch_size,
#             mirror_axes,
#         ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

#         # training pipeline
#         tr_transforms = self.get_training_transforms(
#             patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
#             use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
#             is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
#             regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
#             ignore_label=self.label_manager.ignore_label)

#         # validation pipeline
#         val_transforms = self.get_validation_transforms(deep_supervision_scales,
#                                                         is_cascaded=self.is_cascaded,
#                                                         foreground_labels=self.label_manager.foreground_labels,
#                                                         regions=self.label_manager.foreground_regions if
#                                                         self.label_manager.has_regions else None,
#                                                         ignore_label=self.label_manager.ignore_label)

#         dataset_tr, dataset_val = self.get_tr_and_val_datasets()

#         dl_tr = nnUNetDataLoader3DSPARSE(dataset_tr, self.batch_size,
#                                  initial_patch_size,
#                                  self.configuration_manager.patch_size,
#                                  self.label_manager,
#                                  oversample_foreground_percent=self.oversample_foreground_percent,
#                                  sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
#                                  probabilistic_oversampling=self.probabilistic_oversampling)

#         dl_val = nnUNetDataLoader3DSPARSE(dataset_tr, self.batch_size,
#                                   self.configuration_manager.patch_size,
#                                   self.configuration_manager.patch_size,
#                                   self.label_manager,
#                                   oversample_foreground_percent=self.oversample_foreground_percent,
#                                   sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
#                                   probabilistic_oversampling=self.probabilistic_oversampling)

#         allowed_num_processes = get_allowed_n_proc_DA()
#         if allowed_num_processes == 0:
#             mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
#             mt_gen_val = SingleThreadedAugmenter(dl_val, None)
#         else:
#             mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
#                                                         num_processes=allowed_num_processes,
#                                                         num_cached=max(6, allowed_num_processes // 2), seeds=None,
#                                                         pin_memory=self.device.type == 'cuda', wait_time=0.002)
#             mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
#                                                       transform=None, num_processes=max(1, allowed_num_processes // 2),
#                                                       num_cached=max(3, allowed_num_processes // 4), seeds=None,
#                                                       pin_memory=self.device.type == 'cuda',
#                                                       wait_time=0.002)
        

#         # # let's get this party started
#         _ = next(mt_gen_train)
#         _ = next(mt_gen_val)

#         return mt_gen_train, mt_gen_val


class nnUNetDataLoader3DSPARSE(nnUNetDataLoader):

    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, sampling_probabilities, pad_sides, probabilistic_oversampling, transforms)
        
        #self.idx = np.argsort([s.F['imagename'] for s in self.data_load])
        #self.data_sort = [self.data_load[i] for i in self.idx]

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

        # BF
        self.select_only_labeled_train = True
        self.select_only_labeled_valid = False
        self.testx=[]

    def generate_train_batch(self):

        # BF: TODO
        # Call function to load segmentation, perform np.where to get indices, create dict to map index to class_loaction['-10']
        # self.annotated_classes_key=[-10]
        # do not force force_fg = True

        #start_time = time.time()

        # mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
        # mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
        # data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        #seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        #seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        #print('self.indices123', self.indices)

        #print('generate_train_batch233')
        if self.select_only_labeled_train:
            #selected_class = self.annotated_classes_key
            #print('selected_class123', selected_class)
            selected_keys=[]
            while(len(selected_keys)<self.batch_size):
                index = np.random.choice(self.indices, 1, replace=True, p=self.sampling_probabilities)[0]
                ret = self._data[index]
                kmax = max(self.annotated_classes_key)
                #print('kmax123', kmax)
                ks = [k for k in range(1,kmax+1)]
                random.shuffle(ks)
                for k in ks:
                    if len(ret[3]['class_locations'][k]) > 0:
                        selected_keys.append(index)
                        break

                # class_locations = ret[3]['class_locations']
                # eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                # if len(eligible_classes_or_regions) > 1:
                # #if len(ret[3]['class_locations'][selected_class]) > 0:
                #     #print('cl345', ret[3]['class_locations'][selected_class])
                #     print('index123', index)
                #     selected_keys.append(index)
                #     if index not in self.testx:
                #         self.testx.append(index)
                #         print('testx', len(self.testx))
                #     break

                #import random
                #kmax = max(self.annotated_classes_key)
                #ks = [k for k in range(1,kmax+1)]
                #print('sampling_probabilities123', self.sampling_probabilities)
                #print('ks123', ks)
                #print('kmax', kmax)
                #print('ret123', ret[3]['class_locations'])
                #random.shuffle(ks)
                #for k in ks:
                #    if len(ret[3]['class_locations'][k]) > 0:
                #        selected_keys.append(index)
                #        break

        # print('generate_train_batch233')
        # if self.select_only_labeled_train:
        #     selected_class = self.annotated_classes_key
        #     #print('selected_class123', selected_class)
        #     selected_keys=[]
        #     while(len(selected_keys)<self.batch_size):
        #         index = np.random.choice(self.indices, 1, replace=True, p=self.sampling_probabilities)[0]
        #         ret = self._data[index]
        #         import random
        #         kmax = max(self.annotated_classes_key)
        #         ks = [k for k in range(1,kmax+1)]
        #         print('sampling_probabilities123', self.sampling_probabilities)
        #         #print('ks123', ks)
        #         #print('kmax', kmax)
        #         #print('ret123', ret[3]['class_locations'])
        #         random.shuffle(ks)
        #         for k in ks:
        #             if len(ret[3]['class_locations'][k]) > 0:
        #                 selected_keys.append(index)
        #                 break

        # else:
        #     selected_keys = self.get_indices()

        #print('infinite789', self.infinite)
        #selected_keys = self.get_indices()

        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        #print('self.data_shape018', self.data_shape)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            #print('j123', j, i)

            # !!! BF
            #force_fg = True
            #print('force_fg123', force_fg)

            #selected_keys = self.get_indices()
            #print('selected_key', i)

            # !!!
            #force_fg = True

            data, seg, seg_prev, properties = self._data.load_case(i)

            #print('load_case678', force_fg)

            #print('seg123', seg.shape, np.unique(seg))

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped
            #print('here02')

            #print('bbox_crop', bbox)
            #print('seg_cropped', seg_cropped.shape, np.unique(seg_cropped))

        #print('data_all02', data_all.shape)
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        #print("Time01", (time.time() - start_time))

        if self.transforms is not None:
            #import time
            #start = time.time()
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    #print('data_all123', data_all.shape)
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            #print('data_all04', data_all)
            #print('here03')
            #print("Time02", (time.time() - start_time))
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
        #print('data_all05', data_all)
        #print('here04')
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        #print('get_bbox456')
        #print('annotated_classes_key', self.annotated_classes_key)
        #print('force_fg1236', force_fg)
        #print('class_locations123', class_locations.keys())
        #csize = []
        #for k in class_locations.keys():
        #    csize.append(len(class_locations[k]))
        #print('csize', csize)


        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            #print('selected_class743', selected_class)
            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                #print('selected_voxel123', selected_voxel)
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
                #print('random123')

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs
    
class nnUNetTrainer_ORGAN_1000(nnUNetTrainer_ORGAN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

class nnUNetTrainer_ORGAN_200(nnUNetTrainer_ORGAN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200

class nnUNetTrainer_ORGAN_300(nnUNetTrainer_ORGAN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300

class nnUNetTrainer_ORGAN_600(nnUNetTrainer_ORGAN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 600

class nnUNetTrainer_ORGAN_5(nnUNetTrainer_ORGAN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5