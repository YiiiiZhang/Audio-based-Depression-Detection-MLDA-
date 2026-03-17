import os
import torch
import torchaudio
from .base_dataset import BaseDepressionDataset

class DepressionMFCCDataset(BaseDepressionDataset):
    def __init__(self, *args, **kwargs):
        # Set default max_duration for MFCC if not explicitly provided
        kwargs.setdefault('max_duration', 30.0)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = self.index[idx]
        user_id = item['user_id']
        filename = item['filename']
        label = item['label']
        
        # === Construct full audio path ===
        audio_path = os.path.join(self.audio_root, user_id, self.audio_type, filename)
        
        # === Load audio ===
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Return zero tensor for corrupted files
            waveform = torch.zeros(1, self.target_sr)
            sr = self.target_sr

        # 1. Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # 2. Convert to mono channel
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0) # Shape: (Time,)

        # 3. Truncate
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]

        # Use torch.long for classification targets, torch.float for regression
        return waveform, torch.tensor(label, dtype=torch.long)


class MFCCCollator:
    """
    Padding Collator for MFCC feature extraction
    """
    def __call__(self, batch):
        waveforms = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths) if lengths else 0
        
        batch_size = len(waveforms)
        padded_waveforms = torch.zeros(batch_size, max_len)
        attention_mask = torch.zeros(batch_size, max_len)
        
        for i, (wav, length) in enumerate(zip(waveforms, lengths)):
            padded_waveforms[i, :length] = wav
            attention_mask[i, :length] = 1.0
            
        return {
            "input_values": padded_waveforms,
            "attention_mask": attention_mask,
            "labels": torch.stack(labels)
        }