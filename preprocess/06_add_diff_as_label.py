import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import sys
# ==========================================================
# 0. Add the project root to sys.path to find the 'utils' folder
# ==========================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # Go up one level to the root directory

if project_root not in sys.path:
    sys.path.append(project_root)
from utils.utils import read_json, save_json

# ==========================================================
# Phase 1: Merge Diff Labels into Extracted Dataset
# ==========================================================
save_path = './data/diff_data.json'
coping_diff = read_json('./data/diff_files/coping_diff.json')
training_diff = read_json('./data/diff_files/training_diff.json')
full_extracted_dataset = read_json('./data/extracted_full_dataset.json')

# Initialize the outermost structure needed
new_data = {
    "Coping": {},
    "Training": {}
}

for subject, data_dict in full_extracted_dataset.items():
    
    # Get the data section for the subject
    data_section = data_dict.get('data', {})
    
    # Process Coping data
    if 'Coping' in data_section:
        if subject not in new_data["Coping"]:
            new_data["Coping"][subject] = {}
            
        for audio_file, audio_info in data_section['Coping'].items():
            # Convert audio ID (e.g., "1.wav" -> "01") to match the diff dictionary format
            audio_id = audio_file.replace('.wav', '').zfill(2)
            
            diff_label = None
            if subject in coping_diff and audio_id in coping_diff[subject]:
                diff_label = coping_diff[subject][audio_id]
                
            # Save path, duration, and the extracted diff label
            new_data["Coping"][subject][audio_file] = {
                "path": audio_info.get("path"),
                "duration": audio_info.get("duration"),
                "label": diff_label
            }

    # Process Training data
    if 'Training' in data_section:
        if subject not in new_data["Training"]:
            new_data["Training"][subject] = {}
            
        for audio_file, audio_info in data_section['Training'].items():
            audio_id = audio_file.replace('.wav', '').zfill(2)
            
            diff_label = None
            if subject in training_diff and audio_id in training_diff[subject]:
                diff_label = training_diff[subject][audio_id]
                
            new_data["Training"][subject][audio_file] = {
                "path": audio_info.get("path"),
                "duration": audio_info.get("duration"),
                "label": diff_label
            }

# Save the intermediate merged dataset
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_json(save_path, new_data)
print(f"Merged dataset saved to {save_path}")

# ==========================================================
# Phase 2: Create Dataframe and Perform Stratified Group K-Fold
# ==========================================================
save_root = './data/datasets/Diff'
os.makedirs(save_root, exist_ok=True)

# 1. Flatten the nested JSON dictionary into a list for easy conversion to a DataFrame
records = []
for task in ['Coping', 'Training']:
    if task not in new_data: continue
    for subject, audios in new_data[task].items():
        for audio_name, info in audios.items():
            label_info = info.get('label')
            
            # Filter out samples with missing labels
            if label_info is None:
                continue 
            
            records.append({
                'task': task,
                'subject': subject,
                'audio': audio_name,
                'path': info.get('path'),
                'duration': info.get('duration'),
                'post_n': label_info['post']['n'],
                'post_p': label_info['post']['p'],
                'diff_n': label_info['diff']['n'],
                'diff_p': label_info['diff']['p']
            })

df = pd.DataFrame(records)
print(f"Total valid samples: {len(df)}")
print(f"Number of independent subjects: {df['subject'].nunique()}")

# 2. To ensure positive/negative samples are as balanced as possible during splitting, 
# discretize continuous diff_n into 3 categories.
def bin_diff(val):
    if val < 0: return 0      # Mood improvement (Decrease)
    elif val == 0: return 1   # No change (Neutral)
    else: return 2            # Mood deterioration (Increase)

df['stratify_label'] = df['diff_n'].apply(bin_diff)

# 3. Set up 5-Fold Cross Validation
# StratifiedGroupKFold ensures:
# a) Data from the same subject stays entirely in Train or Val, preventing crossover (Group constraint)
# b) The distribution of mood changes (stratify_label) in Train/Val is as consistent as possible (Stratified constraint)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

post_cv_data = {}
diff_cv_data = {}

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['stratify_label'], groups=df['subject'])):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # ==========================================
    # Task A: Process POST labels (Min-Max scaling)
    # ==========================================
    # Theoretical maximum is 10, divide directly by 10.0 to scale to [0.0, 1.0]
    post_train = []
    for _, row in train_df.iterrows():
        post_train.append({
            'subject': row['subject'], 'task': row['task'], 'audio': row['audio'],
            'path': row['path'], 'duration': row['duration'],
            'label': {'n': row['post_n'] / 10.0, 'p': row['post_p'] / 10.0},
            'original_label': {'n': row['post_n'], 'p': row['post_p']}
        })
        
    post_val = []
    for _, row in val_df.iterrows():
        post_val.append({
            'subject': row['subject'], 'task': row['task'], 'audio': row['audio'],
            'path': row['path'], 'duration': row['duration'],
            'label': {'n': row['post_n'] / 10.0, 'p': row['post_p'] / 10.0},
            'original_label': {'n': row['post_n'], 'p': row['post_p']}
        })
        
    post_cv_data[f"fold_{fold}"] = {
        'train': post_train,
        'val': post_val
    }
    
    # ==========================================
    # Task B: Process DIFF labels (Outlier clipping + Z-score standardization)
    # ==========================================
    # Step 1: Clip to [-3, 3] to prevent extreme values from skewing the regression model
    train_df['diff_n_clip'] = train_df['diff_n'].clip(-3, 3)
    train_df['diff_p_clip'] = train_df['diff_p'].clip(-3, 3)
    val_df['diff_n_clip'] = val_df['diff_n'].clip(-3, 3)
    val_df['diff_p_clip'] = val_df['diff_p'].clip(-3, 3)
    
    # Step 2: Calculate mean and std ONLY on the [Training Set]! (Prevents data leakage)
    mean_n, std_n = train_df['diff_n_clip'].mean(), train_df['diff_n_clip'].std()
    mean_p, std_p = train_df['diff_p_clip'].mean(), train_df['diff_p_clip'].std()
    
    # Prevent division by zero if std is 0
    std_n = std_n if std_n > 1e-6 else 1.0
    std_p = std_p if std_p > 1e-6 else 1.0
    
    # Step 3: Apply Z-score standardization: (x - mean) / std
    diff_train = []
    for _, row in train_df.iterrows():
        diff_train.append({
            'subject': row['subject'], 'task': row['task'], 'audio': row['audio'],
            'path': row['path'], 'duration': row['duration'],
            'label': {
                'n': (row['diff_n_clip'] - mean_n) / std_n,
                'p': (row['diff_p_clip'] - mean_p) / std_p
            },
            'original_label': {'n': row['diff_n'], 'p': row['diff_p']}
        })
        
    diff_val = []
    for _, row in val_df.iterrows():
        # Note: The validation set is also scaled using the train set's mean and std
        diff_val.append({
            'subject': row['subject'], 'task': row['task'], 'audio': row['audio'],
            'path': row['path'], 'duration': row['duration'],
            'label': {
                'n': (row['diff_n_clip'] - mean_n) / std_n,
                'p': (row['diff_p_clip'] - mean_p) / std_p
            },
            'original_label': {'n': row['diff_n'], 'p': row['diff_p']}
        })
        
    diff_cv_data[f"fold_{fold}"] = {
        'scaler_params': {
            'mean_n': mean_n, 'std_n': std_n,
            'mean_p': mean_p, 'std_p': std_p
        },
        'train': diff_train,
        'val': diff_val
    }

# Save cross-validation splits
save_json(os.path.join(save_root, 'post_split.json'), post_cv_data)
save_json(os.path.join(save_root, 'diff_cv_splits.json'), diff_cv_data)

print(f"CV splits saved successfully in {save_root}")