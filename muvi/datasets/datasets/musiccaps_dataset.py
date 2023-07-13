from muvi.utils.lib import *
from muvi.datasets.datasets.base_dataset import BaseDataset
from muvi.datasets.datasets.audio_utils import download_clip

from datasets import load_dataset, Audio


class MusicCapsDataset(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split
        self.ds = load_dataset('google/MusicCaps', split=split)
        self.data_dir = data_dir
        self.resample_rate = processor.sampling_rate
        self.processor = processor
        
        def process_data(example):
            outfile_path = str(data_dir / f"{example['ytid']}.wav")
            status = True
            if not os.path.exists(outfile_path):
                status = False
                status, log = download_clip(
                    example['ytid'],
                    outfile_path,
                    example['start_s'],
                    example['end_s'],
                )

            example['audio'] = outfile_path
            example['download_status'] = status
            return example
        
        self.ds = self.ds.map(
                              process_data,
                              keep_in_memory=False
                             ).cast_column('audio', Audio(sampling_rate=self.resample_rate))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw_audio = self.ds[idx]['audio']['array']
        audio = self.audio_processor(raw_audio, 
                                     sampling_rate=self.resample_rate, 
                                     return_tensors="pt")
        txt = list(self.ds[idx]['caption'])

        return {'audio': audio, 'text_input': txt}

    
