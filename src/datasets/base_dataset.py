import os
import json
from torch.utils.data import Dataset

class BaseDepressionDataset(Dataset):
    """
    Base Dataset class for parsing structured JSON split files.
    Compatible with two JSON formats:
    1. Complex format (Dict): {user_id: {filename: {info}}}
    2. Simple format (List): {user_id: [filename1, filename2]}
    """
    def __init__(self, 
                 json_path: str,           # Path to the split JSON file
                 audio_root: str,          # Root directory of audio data
                 audio_type: str = "Coping", # Audio task type folder name
                 split: str = "train",       # Data split ('train' or 'val')
                 label_type: str = "is_depression", # Target label dimension
                 target_sr: int = 16000, 
                 max_duration: float = 30.0):
        
        self.audio_root = audio_root
        self.audio_type = audio_type
        self.target_sr = target_sr
        self.max_length = int(max_duration * target_sr)
        
        # 1. Load JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            full_structure = json.load(f)
            
        # 2. Locate the specific Split and Label Type
        if split not in full_structure:
            raise ValueError(f"Split '{split}' not found in JSON.")
        
        split_data = full_structure[split]
        
        if label_type not in split_data:
            print(f"Warning: Label type '{label_type}' not found in split '{split}'.")
            target_data = {}
        else:
            target_data = split_data[label_type] 

        # 3. Flatten the data index
        self.index = []
        
        # Iterate through labels (e.g., "0", "1")
        for label_str, users_dict in target_data.items():
            try:
                # Convert label to integer (Modify this to float if handling regression)
                label = int(label_str)
            except ValueError:
                continue # Skip non-numeric keys
            
            # Iterate through users
            for user_id, user_data in users_dict.items():
                
                # === Core Modification: Support both List and Dict structures ===
                if isinstance(user_data, list):
                    # Format: [ "1.wav", "2.wav" ]
                    file_iterator = user_data
                elif isinstance(user_data, dict):
                    # Format: { "1.wav": {...}, "2.wav": {...} }
                    file_iterator = user_data.keys()
                else:
                    print(f"Warning: Unknown data format for user {user_id}")
                    continue
                
                # Iterate through files
                for filename in file_iterator:
                    self.index.append({
                        "user_id": user_id,
                        "filename": filename,
                        "label": label
                    })
                    
        print(f"[{split.upper()}] Loaded {len(self.index)} samples for label '{label_type}'")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement __getitem__ method.")