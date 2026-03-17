import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils.utils import save_json

class Trainer:
    """
    A highly decoupled training engine for PyTorch models.
    """
    def __init__(self, model, optimizer, criterion, device, model_type="mfcc", grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_type = model_type
        self.grad_clip = grad_clip

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        preds, targets = [], []
        
        for batch in loader:
            input_values = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            if input_values.is_floating_point():
                input_values.requires_grad_(True)

            self.optimizer.zero_grad()
            
            if self.model_type == "mfcc":
                outputs = self.model(input_values, attention_mask=attention_mask)
            else: # wav2vec
                outputs = self.model(input_values=input_values)
                
            # Handle different output formats (dict, HuggingFace output, or raw tensor)
            logits = outputs['logits'] if isinstance(outputs, dict) else (outputs.logits if hasattr(outputs, 'logits') else outputs)
                
            loss = self.criterion(logits, labels)
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item() * input_values.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        return avg_loss, acc, f1

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in loader:
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                if self.model_type == "mfcc":
                    outputs = self.model(input_values, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_values=input_values)

                logits = outputs['logits'] if isinstance(outputs, dict) else (outputs.logits if hasattr(outputs, 'logits') else outputs)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * input_values.size(0)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        return avg_loss, acc, f1

    def fit(self, train_loader, val_loader, epochs, save_dir, run_name, save_best=False):
        """
        Execute the full training loop for given epochs.
        """
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        best_f1, best_acc = 0.0, 0.0
        
        for epoch in range(1, epochs + 1):
            t_loss, t_acc, t_f1 = self.train_one_epoch(train_loader)
            v_loss, v_acc, v_f1 = self.evaluate(val_loader)
            
            print(f"    [{run_name}] E{epoch:02d} | Tr_F1={t_f1:.4f} | Val_F1={v_f1:.4f} Val_Acc={v_acc:.4f}")
            
            history["epoch"].append(epoch)
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_f1"].append(v_f1)
            history["val_acc"].append(v_acc)
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                best_acc = v_acc
                if save_best:
                    torch.save(self.model.state_dict(), os.path.join(save_dir, f"{run_name}_best_model.pth"))
                    
        history["best_f1"] = best_f1
        history["best_acc"] = best_acc
        save_json(os.path.join(save_dir, f"{run_name}_log.json"), history)
        
        return best_f1, best_acc