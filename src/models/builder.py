from transformers import Wav2Vec2Processor
from .mfcc_net import MFCCClassifier
from .wav2vec_net import DepressionClassifier

def build_model_and_processor(model_type, num_labels=2):
    """
    Factory method to instantiate models based on type.
    """
    processor = None
    if model_type == "mfcc":
        model = MFCCClassifier(
            sample_rate=16000, 
            n_mfcc=40, 
            hidden_size=256, 
            num_layers=2, 
            dropout=0.4, 
            bidirectional=True
        )
    elif model_type == "wav2vec":
        model_name = "facebook/wav2vec2-base"
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir='./Model/wav2vec2_checkpoints')
        model = DepressionClassifier(
            model_name_or_path=model_name, 
            num_labels=num_labels, 
            pooling_mode="mean"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model, processor