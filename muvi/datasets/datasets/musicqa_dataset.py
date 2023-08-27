from muvi.utils.lib import *
from muvi.datasets.datasets.base_dataset import BaseDataset
from muvi.datasets.datasets.audio_utils import download_clip

from datasets import load_dataset, load_from_disk
import os
import torch
import numpy as np
import json

class MusicQADataset(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split # split is in {musiccaps_pretraining, mtt_finetuning, mtg_evaluation}
        self.data_dir = data_dir #music_data
        self.resample_rate = processor.sampling_rate
        self.processor = processor
        
        data_path = os.path.join(data_dir, 'MusicQA', f'musicqa_{split}')
        self.ds = load_from_disk(data_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        audio = item['Music']['array']
        instruction = [item['Question']]
        txt = [item['Answer']]

        return {'audio': audio, 'text_input': txt, 'instruction_input': instruction}

    def collater(self, samples):
        #padding to max length in a batch
        audios = [s['audio'] for s in samples]
        audio_sizes = [len(s['audio']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [" ".join(s['text_input']) for s in samples]
        instructions = [" ".join(s['instruction_input']) for s in samples]

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

        return {'audio': collated_audios, 'text_input': txts, 'instruction_input': instructions, 'attention_mask': attn_mask}
