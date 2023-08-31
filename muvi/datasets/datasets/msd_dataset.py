from muvi.utils.lib import *
from muvi.datasets.datasets.base_dataset import BaseDataset

from datasets import load_dataset, Audio
import torchaudio
import os
import torch
import numpy as np
import json
import torchaudio.transforms as T


class MSDDataset(BaseDataset):
    def __init__(self, processor, mp3_file_folder, split, msd="seungheondoh/LP-MusicCaps-MSD"):
        assert(split in ["train", "valid", "test"])
        self.raw_file_folder = mp3_file_folder
        self.msd_dataset = load_dataset(msd)[split]
        self.resample_rate = processor.sampling_rate #music_data
        self.processor = processor


    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, idx):
        data_dict = self.msd_dataset[idx]
        data_path = data_dict["path"]
        filename_path = os.path.join(self.raw_file_folder, data_path)
        caption = data_dict["caption_writing"]
        #try:
        #        audio_metadata = torchaudio.info(filename_path)
        #        sampling_rate = audio_metadata.sampling_rate
        #except:
        #        sampling_rate = 22050
        #audio_dict = {"path": filename_path, "bytes":None}
        #feature_dict = Audio(sampling_rate=22050).decode_example(audio_dict)
        #npy_features = feature_dict["array"]
        #res_dict = {}
        try:
            waveform, sampling_rate = torchaudio.backend.soundfile_backend.load(filename_path)
            audio_tensor = torch.mean(waveform, dim=0)
            resampler = T.Resampler(sampling_rate, self.resample_rate)
            audio_input = resampler(audio_tensor.float())

            audio = self.processor(audio_input,
                                   sampling_rate=self.resample_rate,
                                   return_tensors="pt")['input_values'][0]
            res_dict["audio"] = audio
            res_dict["text_input"] = [caption]
            return res_dict
        except:
            return self.__getitem__(0)


    def collater(self, samples):
        audios = [s['audio'] for s in samples]
        audio_sizes = [len(s['audio']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [s['text_input'] for s in samples]

        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        attn_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(True)
        )

        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            else: #diff < 0
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                attn_mask[i, diff:] = False

        attn_mask = attn_mask.int()

        return {'audio': collated_audios, 'text_input': txts, 'attention_mask': attn_mask}
