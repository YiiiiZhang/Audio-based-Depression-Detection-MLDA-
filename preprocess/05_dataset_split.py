import json
import random
import os
import sys
# ==========================================================
# 0. Add the project root to sys.path to find the 'utils' folder
# ==========================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # Go up one level to the root directory

if project_root not in sys.path:
    sys.path.append(project_root)
from utils.utils import read_json

def stratified_greedy_kfold(grouped_users, k_folds):
    """
    Helper function: Performs a greedy K-fold split on the given user dictionary {"0": [...], "1": [...]}.
    It ensures the total number of audio files in each fold is as balanced as possible, 
    while maintaining consistent class proportions (stratified).
    """
    # Create K buckets
    buckets = [{"0": [], "1": []} for _ in range(k_folds)]
    
    for val_key in ["0", "1"]:
        users = grouped_users[val_key]
        # Sort users by their audio count in descending order
        users_sorted = sorted(users, key=lambda x: x['count'], reverse=True)
        # Track the total number of files for the current class across the K buckets
        fold_counts = [0] * k_folds
        
        for user in users_sorted:
            # Find the bucket with the minimum data count currently
            min_idx = fold_counts.index(min(fold_counts))
            buckets[min_idx][val_key].append(user)
            fold_counts[min_idx] += user['count']
            
    return buckets

def users_to_dict(users_grouped):
    """
    Helper function: Converts a list containing user objects into the target output format: 
    {"0": {"uid": [files...]}, "1": {...}}
    """
    out = {"0": {}, "1": {}}
    for val_key in ["0", "1"]:
        for user in users_grouped[val_key]:
            out[val_key][user['uid']] = user['files']
    return out

def generate_nested_cv_splits_multi_labels(data, audio_type, k_outer=5, k_inner=3, save_path=None, seed=42):
    """
    Generates Nested Cross-Validation (Nested CV) data split indices based on UserId.
    
    Args:
        data (dict): Original dataset
        audio_type (str): Type of audio (e.g., 'Coping', 'Training')
        k_outer (int): Number of outer folds (used for model evaluation), default 5
        k_inner (int): Number of inner folds (used for hyperparameter tuning), default 3
        save_path (str): File path to save the generated splits
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    TARGET_LABEL_KEYS = [
        "is_depression", "is_HRSD", "is_retardation", 
        "is_insomnia", "is_agitation", "is_weight_loss"
    ]

    # Initialize output structure: output['label_name'] = [ outer_fold_1, outer_fold_2... ]
    output = {k: [] for k in TARGET_LABEL_KEYS}
    
    for label_key in TARGET_LABEL_KEYS:
        # 1. Data cleaning: Extract users with valid labels (0 and 1)
        grouped_users = {"0": [], "1": []}
        
        for user_id, user_info in data.items():
            if 'data' not in user_info or audio_type not in user_info['data']: continue
            
            wav_list = user_info['data'][audio_type]
            count = len(wav_list)
            if count == 0: continue

            if 'label' in user_info and label_key in user_info['label']:
                label_str = str(user_info['label'][label_key])
                if label_str not in ["0", "1"]: continue
                
                grouped_users[label_str].append({
                    'uid': user_id, 'files': wav_list, 'count': count
                })
        
        # 2. Outer Split
        # Divide all data into k_outer buckets
        outer_buckets = stratified_greedy_kfold(grouped_users, k_outer)
        
        for outer_idx in range(k_outer):
            # The current bucket acts as the Outer Test set
            outer_test_users = outer_buckets[outer_idx]
            
            # Merge the remaining buckets to form the Outer Train set
            outer_train_users = {"0": [], "1": []}
            for i in range(k_outer):
                if i != outer_idx:
                    outer_train_users["0"].extend(outer_buckets[i]["0"])
                    outer_train_users["1"].extend(outer_buckets[i]["1"])
                    
            # 3. Inner Split
            # Take the newly merged outer train set and perform k_inner splits on it
            inner_buckets = stratified_greedy_kfold(outer_train_users, k_inner)
            
            inner_splits_formatted = []
            for inner_idx in range(k_inner):
                # The current inner bucket acts as the Inner Validation set (for hyperparameter tuning)
                inner_val_users = inner_buckets[inner_idx]
                
                # Merge the remaining inner buckets to form the Inner Train set
                inner_train_users = {"0": [], "1": []}
                for j in range(k_inner):
                    if j != inner_idx:
                        inner_train_users["0"].extend(inner_buckets[j]["0"])
                        inner_train_users["1"].extend(inner_buckets[j]["1"])
                
                inner_splits_formatted.append({
                    "inner_fold_id": inner_idx + 1,
                    "inner_train": users_to_dict(inner_train_users),
                    "inner_val": users_to_dict(inner_val_users)
                })
            
            # 4. Assemble the data structure for the current Outer Fold
            outer_fold_dict = {
                "outer_fold_id": outer_idx + 1,
                "outer_test": users_to_dict(outer_test_users),
                # [Note]: We do not need to save outer_train, because outer_train equals the sum of all inner splits
                "inner_splits": inner_splits_formatted
            }
            output[label_key].append(outer_fold_dict)

    # Print some statistics to verify the split distribution
    print(f"--- Nested CV Split Completed for {audio_type} ---")
    for l_key in TARGET_LABEL_KEYS:
        if len(output[l_key]) == 0: continue
        
        # Grab the first outer fold just to sample and print counts
        fold1 = output[l_key][0]
        test_0 = sum(len(f) for f in fold1['outer_test']['0'].values())
        test_1 = sum(len(f) for f in fold1['outer_test']['1'].values())
        
        in_train_0 = sum(len(f) for f in fold1['inner_splits'][0]['inner_train']['0'].values())
        in_val_0 = sum(len(f) for f in fold1['inner_splits'][0]['inner_val']['0'].values())
        
        print(f"[{l_key}] Outer Fold 1 Example:")
        print(f"  Outer Test : {test_0} (cls 0), {test_1} (cls 1)")
        print(f"  Inner Folds: Contains {k_inner} inner splits (e.g., Inner 1 has Train_0={in_train_0}, Val_0={in_val_0})")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"Nested CV indices saved to {save_path}\n")
    
    return output


# ===== Execution Block =====
if __name__ == "__main__":
    json_path = "./data/extracted_full_dataset.json"
    full_data = read_json(json_path)
    
    for audio_type in ["Coping", "Training"]:
        # Ensure the directory exists before saving
        save_dir = f"./data/datasets/{audio_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = f"{save_dir}/{audio_type}_Split.json"
        generate_nested_cv_splits_multi_labels(data=full_data, audio_type=audio_type, save_path=save_path)