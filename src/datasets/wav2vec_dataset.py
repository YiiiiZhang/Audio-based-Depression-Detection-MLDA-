import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from .base_dataset import BaseDepressionDataset

class DepressionWav2VecDataset(BaseDepressionDataset):
    def __init__(self, *args, **kwargs):
        # Set default max_duration for Wav2Vec if not explicitly provided
        kwargs.setdefault('max_duration', 10.0)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = self.index[idx]
        
        # Construct full audio path
        audio_path = os.path.join(self.audio_root, item['user_id'], self.audio_type, item['filename'])
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            waveform = torch.zeros(1, self.target_sr)
            sr = self.target_sr

        # 1. Resample (Wav2Vec2 strictly requires 16kHz)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # 2. Convert to mono channel
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 3. Flatten (1, T) -> (T,)
        waveform = waveform.squeeze()

        # 4. Truncate (keep only the first max_duration seconds)
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]

        # Return numpy array (required by HF Processor) and label
        return waveform.numpy(), item['label']


class Wav2VecCollator:
    """
    Padding and Normalization Collator using HuggingFace Processor
    """
    def __init__(self, processor_name="facebook/wav2vec2-base", processor=None):
        if processor:
            self.processor = processor
        else:
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
            except:
                # Fallback path if running offline
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __call__(self, batch):
        # batch: List of (waveform_numpy, label)
        raw_speech = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Core: Use Processor for Padding and Normalization
        # sampling_rate=16000 is mandatory
        batch_out = self.processor(
            raw_speech, 
            sampling_rate=16000, 
            padding=True, 
            return_tensors="pt"
        )
        
        # Return dictionary mapping model input requirements
        return {
            "input_values": batch_out.input_values,  # Shape: (Batch, Time)
            # "attention_mask": batch_out.attention_mask, 
            "labels": torch.tensor(labels, dtype=torch.long)
        }