from .mfcc_dataset import DepressionMFCCDataset, MFCCCollator
from .wav2vec_dataset import DepressionWav2VecDataset, Wav2VecCollator

def build_dataset_and_collator(model_type, json_path, audio_root, audio_type, split, label_type, processor=None):
    """
    Factory method to instantiate datasets and collators.
    """
    if model_type == "mfcc":
        dataset = DepressionMFCCDataset(
            json_path=json_path, audio_root=audio_root, audio_type=audio_type, 
            split=split, label_type=label_type, max_duration=60.0
        )
        collator = MFCCCollator()
    elif model_type == "wav2vec":
        dataset = DepressionWav2VecDataset(
            json_path=json_path, audio_root=audio_root, audio_type=audio_type, 
            split=split, label_type=label_type, max_duration=10.0
        )
        collator = Wav2VecCollator(processor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return dataset, collator