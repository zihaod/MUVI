import os
import logging
import warnings
import torch
from transformers import Wav2Vec2FeatureExtractor

from muvi.common.registry import registry
from muvi.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from muvi.datasets.datasets.musiccaps_dataset import MusicCapsDataset


@registry.register_builder("musiccaps")
class MusicCapsBuilder(BaseDatasetBuilder):
    train_dataset_cls = MusicCapsDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/musiccaps/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split, 
            data_dir=self.config.data_dir, 
        )
 

        return datasets



@registry.register_builder("musicqa")
class MusicQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MusicQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/musicqa/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        #get processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.processor, trust_remote_code=True)


        # create datasets
        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            processor=processor,
            split=self.config.split, 
            data_dir=self.config.data_dir, 
        )
 

        return datasets
