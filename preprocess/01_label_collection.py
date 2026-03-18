import os
import json
import sys
from tqdm import tqdm

#Import the modules in the utils folder
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.audio_process import get_duration
from utils.utils import save_json
from utils.label_process import build_full_dataset

# ==========================================================
# 1. Dynamically load paths from configs/base_env.json

config_path = os.path.join(project_root, 'configs', 'base_env.json')
# Read the JSON configuration file
with open(config_path, 'r', encoding='utf-8') as f:
    env_config = json.load(f)

# Replace the hardcoded path with the parameter from the JSON file
rp = env_config.get("RAW_VIDEO_DIR")
# 2. Process data and generate dataset
# ==========================================================
dataset_files = './data/full_dataset.json'
all_user_list = os.listdir(rp)
all_data = {}

# Initial dataset: Users containing t2 audios
for u_id in tqdm(all_user_list, desc="Processing Users"):
    # Skip invalid or non-user data files/folders
    if u_id in ['new_data', 'NotProcessed_Data_updated.csv']: 
        continue
    
    user_path = os.path.join(rp, u_id)
    
    # Ensure the path is actually a directory before attempting to list its contents
    if not os.path.isdir(user_path):
        continue
        
    file_list = os.listdir(user_path)
    
    for file in file_list:
        # Check for t2 mp4 files
        if 't2' in file and file.endswith('.mp4'):
            video_path = os.path.join(user_path, file)
            duration = get_duration(video_path)
            all_data[u_id] = {
                'data': {
                    'Raw_data': {
                        'path': video_path,
                        'duration': duration
                    }
                }
            }

# Save the final structured dictionary into a JSON file
save_json(
    dataset_files, 
    build_full_dataset(all_data, '../data/labels/20251105_d02_questionnaires_app.xlsx')
)