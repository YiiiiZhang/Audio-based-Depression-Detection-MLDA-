import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from utils.utils import read_json, save_json
from src.trainer import Trainer
from src.models.builder import build_model_and_processor
from src.datasets.builder import build_dataset_and_collator

# ====================================================
# Helper Functions
# ====================================================
def get_outer_train(inner_split_0):
    """
    Reconstruct the outer training set by merging inner train and inner val.
    """
    outer_train = {"0": {}, "1": {}}
    for val_key in ["0", "1"]:
        outer_train[val_key].update(inner_split_0["inner_train"].get(val_key, {}))
        outer_train[val_key].update(inner_split_0["inner_val"].get(val_key, {}))
    return outer_train

def get_cb_class_weights(counts, beta=0.9999):
    """
    Calculate Class-Balanced (CB) weights to handle imbalanced datasets.
    """
    counts = np.array(counts)
    if np.sum(counts) == 0: 
        return torch.tensor([1.0, 1.0])
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (np.array(effective_num) + 1e-9)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

def run_single_fold(train_dict, val_dict, label_type, audio_root,args, device, lr, save_dir, run_name):
    """
    Encapsulated single training run (Data -> Model -> Train -> Cleanup).
    Memory isolation is strictly maintained here to prevent Out-Of-Memory (OOM) errors.
    """
    # 1. Temporarily save the data dictionary required for the current Fold
    temp_json_path = os.path.join(save_dir, f"temp_{run_name}.json")
    save_json(temp_json_path, {"train": {label_type: train_dict}, "val": {label_type: val_dict}})

    batch_size = max(4, args.batch_size // 4) if args.model == "wav2vec" else args.batch_size

    try:
        # 2. Build model, processor, and datasets using Factory methods
        model, processor = build_model_and_processor(args.model)
        model = model.to(device)
        
        train_set, collator = build_dataset_and_collator(
            args.model, temp_json_path, audio_root, args.audio_type, "train", label_type, processor
        )
        val_set, _ = build_dataset_and_collator(
            args.model, temp_json_path, audio_root, args.audio_type, "val", label_type, processor
        )
        
        if len(train_set) == 0 or len(val_set) == 0:
            return 0.0, 0.0

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4)

        # 3. Compute class weights and initialize optimizer
        labels = [item['label'] for item in train_set.index]
        counts = [Counter(labels).get(0, 0), Counter(labels).get(1, 0)]
        class_weights = get_cb_class_weights(counts).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # 4. Initialize Trainer and execute training loop
        trainer = Trainer(model, optimizer, criterion, device, model_type=args.model)
        
        # Only save the best model weights during Outer Evaluation
        save_best = "outer" in run_name
        best_f1, best_acc = trainer.fit(train_loader, val_loader, args.epochs, save_dir, run_name, save_best=save_best)

    except Exception as e:
        print(f"[{run_name}] Error during execution: {e}")
        return 0.0, 0.0

    finally:
        # 5. [CRITICAL MEMORY MANAGEMENT]: Forcibly release GPU memory occupied by this fold
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        torch.cuda.empty_cache()
    
    return best_f1, best_acc

# ====================================================
# Main Execution Pipeline
# ====================================================
def main(label_type, json_path, save_dir,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)
    all_splits_data = read_json(json_path)
    if label_type not in all_splits_data:
        print(f"Label {label_type} not found in splits. Skipping.")
        return
    
    outer_folds_data = all_splits_data[label_type]
    print(f"\n{'='*15} Task: {label_type} | Nested CV Folds: {len(outer_folds_data)} {'='*15}")
    final_metrics = {"test_f1": [], "test_acc": []}
    
    # --- Outer Cross-Validation Loop ---
    for outer_fold in outer_folds_data:
        outer_id = outer_fold["outer_fold_id"]
        print(f"\n>>> Starting OUTER Fold {outer_id}/{len(outer_folds_data)} <<<")
        fold_save_dir = os.path.join(save_dir, f"outer_fold_{outer_id}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        outer_test = outer_fold["outer_test"]
        inner_splits = outer_fold["inner_splits"]
        outer_train = get_outer_train(inner_splits[0]) 

        # --- Inner Cross-Validation Loop (Hyperparameter Search) ---
        lr_candidates = [args.lr, args.lr * 0.1]
        best_lr, best_inner_score = args.lr, -1
        
        for lr in lr_candidates:
            inner_f1_scores = []
            for inner_fold in inner_splits:
                val_f1, _ = run_single_fold(
                    train_dict=inner_fold["inner_train"], val_dict=inner_fold["inner_val"], 
                    label_type=label_type, args=args, device=device, lr=lr, 
                    save_dir=fold_save_dir, run_name=f"in_fold{inner_fold['inner_fold_id']}_lr_{lr}"
                )
                inner_f1_scores.append(val_f1)
                
            avg_inner_f1 = np.mean(inner_f1_scores)
            if avg_inner_f1 > best_inner_score:
                best_inner_score, best_lr = avg_inner_f1, lr

        # --- Outer Final Evaluation (Test) ---
        test_f1, test_acc = run_single_fold(
            train_dict=outer_train, val_dict=outer_test, 
            label_type=label_type, args=args, device=device, lr=best_lr, 
            save_dir=fold_save_dir, run_name="outer_eval"
        )
        
        final_metrics["test_f1"].append(test_f1)
        final_metrics["test_acc"].append(test_acc)
        
    # --- Print Generalization Summary ---
    mean_f1, std_f1 = np.mean(final_metrics["test_f1"]), np.std(final_metrics["test_f1"])
    print(f"\n[Final Results for {label_type}] F1: {mean_f1:.4f} ± {std_f1:.4f} | Acc: {np.mean(final_metrics['test_acc']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wav2vec", choices=["mfcc", "wav2vec"])
    parser.add_argument("--audio_type", type=str, default="Training", choices=["Training", "Coping"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    audio_root = read_json('.configs/base_env.json').get("FINAL_AUDIO_DIR",None)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    for label in ['is_depression', 'is_agitation', 'is_retardation', 'is_HRSD']:
        json_path = f"./data/datasets/{args.audio_type}/{args.audio_type}_Split.json"
        save_dir = f"./Output/{args.model}/{args.audio_type}/{label}_nested"
        main(label, json_path, save_dir, args)