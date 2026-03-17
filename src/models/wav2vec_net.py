import torch
from torch import nn
from transformers import Wav2Vec2Model

def freeze_module(module: nn.Module):
    """
    Freeze module parameters.
    """
    for p in module.parameters():
        p.requires_grad = False


class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DepressionClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_labels=2, pooling_mode="mean"):
        super().__init__()
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name_or_path)
        self.config = self.wav2vec2.config
        
        # Disable gradient checkpointing to save memory
        self.wav2vec2.config.gradient_checkpointing = False 
        self.config.num_labels = num_labels
        self.config.final_dropout = 0.1  # Can be adjusted as needed
        self.pooling_mode = pooling_mode
        
        # Freeze feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Classification Head
        self.classifier = Wav2Vec2ClassificationHead(self.config)

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError("Pooling mode must be 'mean' or 'max'")

    def forward(self, input_values, labels=None):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Aggregate features (Pooling)
        pooled_output = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}


#------------------------- ------------------------- ------------------------- 
# Negative & positive regression Head
#------------------------- ------------------------- ------------------------- 

class Wav2Vec2RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        # Note here: the regression task only outputs 1 continuous value
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DepressionRegressorWav2Vec(nn.Module):
    def __init__(self, model_name_or_path, pooling_mode="mean"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name_or_path)
        self.config = self.wav2vec2.config
        
        # 1. Enable gradient checkpointing (must be kept to prevent OOM)
        self.wav2vec2.gradient_checkpointing_enable()
        
        # 2. Freeze bottom-layer feature extractor (must be kept)
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Regression head related configuration
        self.config.final_dropout = 0.1 
        self.pooling_mode = pooling_mode
        self.regressor = Wav2Vec2RegressionHead(self.config)

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError("Pooling mode must be 'mean' or 'max'")

    def forward(self, input_values, labels=None):
        if self.training:
            input_values.requires_grad_(True)
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Aggregate features (Pooling)
        pooled_output = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        # Regression prediction (B, 1) -> (B,)
        preds = self.regressor(pooled_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            # Ensure labels type is float and shapes match
            labels = labels.to(preds.dtype).view(-1)
            # Use MSELoss to calculate regression error
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)
            
        return {"loss": loss, "preds": preds}