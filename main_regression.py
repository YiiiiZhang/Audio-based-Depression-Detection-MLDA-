import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor  
import torchaudio
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.utils import read_json, save_json
from src.models.mfcc_net import MFCCRegressor
from src.models.wav2vec_net import DepressionRegressorWav2Vec

# ====================================================
# 1. Utility Functions
# ====================================================
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ====================================================
# 2. Dataset & Collator (Specific to Regression)
# ====================================================
class RegressionAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, audio_root, dim='n', processor=None, is_wav2vec=False, max_duration=15.0):
        self.audio_root = audio_root
        self.data_list = data_list
        self.dim = dim
        self.processor = processor
        self.is_wav2vec = is_wav2vec
        self.max_duration = max_duration
        self.target_sr = 16000

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_path = os.path.join(self.audio_root, item['subject'], item['task'], item['path']) 
        label = float(item['label'][self.dim])

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            waveform, sr = torch.zeros(1, self.target_sr), self.target_sr

        # Resampling
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        waveform = waveform.squeeze(0)
        
        # Truncation
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.target_sr)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
                
        # Feature Extraction
        if self.is_wav2vec and self.processor is not None:
            input_values = self.processor(waveform, sampling_rate=self.target_sr, return_tensors="pt").input_values.squeeze(0)
        else:
            input_values = waveform

        return {"input_values": input_values, "label": label}

class RegressionCollator:
    def __init__(self, is_wav2vec=False, processor=None):
        self.is_wav2vec = is_wav2vec
        self.processor = processor

    def __call__(self, features):
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)

        if self.is_wav2vec and self.processor is not None:
            input_features = [{"input_values": f["input_values"]} for f in features]
            batch = self.processor.pad(
                input_features, padding=True, return_attention_mask=True, return_tensors="pt"
            )
            return {
                "input_values": batch.input_values, 
                "attention_mask": batch.get("attention_mask", None), 
                "labels": labels
            }
        else:
            input_values = [f["input_values"] for f in features]
            input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
            lengths = torch.tensor([len(v) for v in input_values])
            max_len = input_values_padded.size(1)
            attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
            
            return {
                "input_values": input_values_padded, 
                "attention_mask": attention_mask.float(), 
                "labels": labels
            }

# ====================================================
# 3. Training & Evaluation Engine
# ====================================================
class RegressionTrainer:
    """
    Decoupled Engine for Continuous Value Prediction Tasks (Regression)
    """
    def __init__(self, model, optimizer, device, is_mfcc=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.is_mfcc = is_mfcc

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.is_mfcc:
                outputs = self.model(input_values, attention_mask=attention_mask, labels=labels)
            else:
                outputs = self.model(input_values, labels=labels)
                
            loss = outputs["loss"]
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, task, mean_val=0.0, std_val=1.0):
        self.model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.is_mfcc:
                    outputs = self.model(input_values, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = self.model(input_values, labels=labels)
                    
                total_val_loss += outputs["loss"].item()
                all_preds.extend(outputs["preds"].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = total_val_loss / len(dataloader)
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        
        # === Reverse Scaling to acquire actual psychological scores ===
        if task == "diff":
            real_preds = all_preds * std_val + mean_val
            real_labels = all_labels * std_val + mean_val
        elif task == "post":
            real_preds = all_preds * 10.0
            real_labels = all_labels * 10.0

        mae = mean_absolute_error(real_labels, real_preds)
        mse = mean_squared_error(real_labels, real_preds)
        pcc = 0.0 if np.std(real_preds) == 0 or np.std(real_labels) == 0 else pearsonr(real_labels, real_preds)[0]
            
        return val_loss, mae, mse, pcc

    def fit(self, train_loader, val_loader, epochs, save_dir, run_name, task, mean_val, std_val):
        """
        Execute the full training loop for given epochs.
        """
        best_val_mae = float('inf')
        best_metrics = {}
        history = {"epochs": []}
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae, val_mse, val_pcc = self.evaluate(val_loader, task, mean_val, std_val)
            
            print(f"    [{run_name}] E{epoch:02d} | Tr_Loss: {train_loss:.4f} | Val_MAE: {val_mae:.4f} | Val_PCC: {val_pcc:.4f}")
            
            history["epochs"].append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_mae": val_mae, "val_mse": val_mse, "val_pcc": val_pcc
            })
            
            # Update best metrics based on MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_metrics = {"mae": val_mae, "mse": val_mse, "pcc": val_pcc, "val_loss": val_loss, "epoch": epoch}
                # Optional: torch.save(self.model.state_dict(), os.path.join(save_dir, f"{run_name}_best.pth"))
                
        history["best_metrics"] = best_metrics
        save_json(os.path.join(save_dir, f"{run_name}_log.json"), history)
        return best_metrics

# ====================================================
# 4. Single Fold Execution (OOM Prevention)
# ====================================================
def run_single_fold(train_list, val_list, dim, args, device, mean_val, std_val, fold_name, save_dir):
    """
    Encapsulated single training run (Data -> Model -> Train -> Cleanup).
    Strict memory isolation to prevent OOM across multiple folds.
    """
    try:
        # 1. Build Processor & Collator
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_name, cache_dir=args.cache_dir) if args.model == "wav2vec" else None
        collator = RegressionCollator(is_wav2vec=(args.model == "wav2vec"), processor=processor)
        
        # 2. Build Datasets & Loaders
        train_dataset = RegressionAudioDataset(train_list, args.audio_root, dim, processor, args.model == "wav2vec", args.max_duration)
        val_dataset = RegressionAudioDataset(val_list, args.audio_root, dim, processor, args.model == "wav2vec", args.max_duration)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
        
        # 3. Build Model & Optimizer
        if args.model == "wav2vec":
            model = DepressionRegressorWav2Vec(args.wav2vec_name, pooling_mode="mean")
        else:
            model = MFCCRegressor()
            
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 4. Initialize Trainer Engine & Fit
        trainer = RegressionTrainer(model, optimizer, device, is_mfcc=(args.model == "mfcc"))
        best_metrics = trainer.fit(train_loader, val_loader, args.epochs, save_dir, fold_name, args.task, mean_val, std_val)

    except Exception as e:
        print(f"[{fold_name}] Error during execution: {e}")
        return {"mae": 0, "mse": 0, "pcc": 0}

    finally:
        # [CRITICAL MEMORY MANAGEMENT]: Forcibly release GPU memory
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        torch.cuda.empty_cache()
        
    return best_metrics

# ====================================================
# 5. Main Execution Pipeline
# ====================================================
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Iterate through specified regression dimensions (e.g., positive diff 'p' or negative diff 'n')
    for dim in ["p", "n"]:
        print(f"\n[{args.model.upper()}] Task: {args.task.upper()} | Dim: {dim.upper()} | Device: {device}")
        
        exp_name = f"{args.model}_{args.task}_{dim}"
        save_path = os.path.join(args.save_dir, exp_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Load splitting strategy
        json_name = f"{args.task}_cv_splits.json"
        data_path = os.path.join(args.data_dir, json_name)
        split_data = read_json(data_path)
        
        all_folds_metrics = []
        
        # Standard K-Fold Cross Validation
        for fold in range(5):
            fold_name = f"fold_{fold}"
            print(f"\n>>> Starting {fold_name.upper()} <<<")
            
            train_list = split_data[fold_name]["train"]
            val_list = split_data[fold_name]["val"]
            
            # Extract Scaler Params for Reverse Scaling
            mean_val, std_val = 0.0, 1.0
            if args.task == "diff":
                scaler_params = split_data[fold_name].get("scaler_params", {})
                mean_val = scaler_params.get(f"mean_{dim}", 0.0)
                std_val = scaler_params.get(f"std_{dim}", 1.0)
                
            # Execute Single Fold (OOM Safe)
            best_metrics = run_single_fold(
                train_list, val_list, dim, args, device, mean_val, std_val, fold_name, save_path
            )
            
            print(f"Fold {fold} Best Val MAE: {best_metrics.get('mae', 0):.4f} (Epoch {best_metrics.get('epoch', 'N/A')})")
            all_folds_metrics.append(best_metrics)
            
        # ================= Cross Validation Summary =================
        fold_maes = [m.get("mae", 0) for m in all_folds_metrics]
        fold_mses = [m.get("mse", 0) for m in all_folds_metrics]
        fold_pccs = [m.get("pcc", 0) for m in all_folds_metrics]
        
        summary_text = (
            f"\n{'*'*40}\n"
            f"Cross Validation Summary ({exp_name}):\n"
            f"Mean MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}\n"
            f"Mean MSE: {np.mean(fold_mses):.4f} ± {np.std(fold_mses):.4f}\n"
            f"Mean PCC: {np.mean(fold_pccs):.4f} ± {np.std(fold_pccs):.4f}\n"
            f"Folds MAE Details: {[round(x, 4) for x in fold_maes]}\n"
            f"{'*'*40}\n"
        )
        print(summary_text)
        
        with open(os.path.join(save_path, "cv_summary.txt"), "w") as f:
            f.write(summary_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Regression Training Pipeline")
    parser.add_argument("--task", type=str, default="diff", choices=["diff", "post"], help="Task type: diff or post")
    parser.add_argument("--audio_root", type=str, default="/home/woody/iwso/iwso192h/MLDA/extracted_data")
    parser.add_argument("--model", type=str, default="wav2vec", choices=["mfcc", "wav2vec"], help="Model architecture")
    parser.add_argument("--data_dir", type=str, default="./data/datasets/Diff/", help="Root dir for split jsons")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Max audio duration in seconds")
    parser.add_argument("--wav2vec_path", type=str, default="facebook/wav2vec2-base", help="Wav2Vec2 pretrained path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./Output/Regression", help="Model and logs save directory")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    main(args)