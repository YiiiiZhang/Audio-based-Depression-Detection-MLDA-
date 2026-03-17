import os
import sys
from tqdm import tqdm

# ==========================================================
# 0. Add the project root to sys.path to find the 'utils' folder
# ==========================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # Go up one level to the root directory

if project_root not in sys.path:
    sys.path.append(project_root)

# Import all necessary utility functions
from utils.utils import save_json, read_json, state_distribution
from utils.audio_process import (
    read_Ei_timerange, 
    extract_Ei_audio_segments, 
    batch_extract_audio_without_silence
)

# ==========================================================
# 1. Dynamically load paths and configurations
# ==========================================================
config_path = os.path.join(project_root, 'configs', 'base_env.json')
BASE_CONFIG = read_json(config_path)

# Ensure these match the keys in your base_env.json exactly
rp_video = BASE_CONFIG.get('RAW_VIDEO_DIR')
save_root = BASE_CONFIG.get('EXTRACTED_AUDIO_DIR')

# Load the main dataset
data_path = './data/full_dataset.json'
data = read_json(data_path)

# ==========================================================
# 2. Setup dictionaries and helper variables
# ==========================================================
# Dictionary mapping user IDs to their app log files for Ei processing
rp_app_logs = './data/app_logs'
app_file_list = os.listdir(rp_app_logs)
app_dict = {fo.split('_')[0]: os.path.join(rp_app_logs, fo) for fo in app_file_list}

# Dictionary handling specific edge cases or manually corrected labels for Coping/Training
err_dict = {
    '984': 'crAdk', '077': None, '742': 'cr', '1006': 'crAdk', 
    '907': 'crAdk', '911': 'crAdk', '954': 'crAdk', '940': 'cr', 
    '932': 'crAdk', '887': 'cr', '760': 'cr', '1040': 'crAdk', 
    '1014': 'cr', '866': 'cr', '1024': 'crAdk', '859': 'cr', 
    '731': 'cr', '987': 'crAdk', '955': 'cr', '1028': 'cr', 
    '861': 'crAdk', '1013': 'cr', '982': 'crAdk', '735': 'cr', 
    '995': 'cr', '1069': 'cr', '1104': 'crAdk', '905': 'crAdk'
}

# ==========================================================
# 3. Unified Processing Loop (Ei + Coping + Training)
# ==========================================================
for k, v in tqdm(data.items(), desc="Extracting All Audio Segments"):
    
    # ------------------------------------------------------
    # Phase A: Process Ei Audio Segments
    # ------------------------------------------------------
    if k in app_dict:
        try:
            # 1. Read time ranges from the app log
            ei_timeranges = read_Ei_timerange(app_dict[k], encoding='utf-8')
            
            # 2. Extract the segments from the raw mp4 file
            mp4_path = v.get('data', {}).get('Raw_data', {}).get('path')
            
            if mp4_path and ei_timeranges:
                path_dict = extract_Ei_audio_segments(
                    minx_key=k,
                    value=ei_timeranges,
                    mp4_path=mp4_path,
                    save_root=save_root,
                    sr=16000,
                    channels=1,
                    audio_fmt="wav"
                )
                # Store the extracted paths back into the dataset
                v['data']['Ei'] = path_dict
        except Exception as e:
            print(f"[{k}] Ei Processing ERROR: {e}")

    # ------------------------------------------------------
    # Phase B: Process Coping and Training Audio Segments
    # ------------------------------------------------------
    try:
        # Create user specific output directory
        k_path = os.path.join(save_root, k)
        os.makedirs(k_path, exist_ok=True)
        
        # Check label conditions: Only 'CR' or 'CR_ADK' and ignore '077'
        if v['label']['type'] in ['CR', 'CR_ADK'] and k not in ['077']:
            
            # Determine correct folder name using err_dict override if applicable
            tyname = v['label']['type'] if k not in err_dict else err_dict[k]
            rp_TC_clips = os.path.join(rp_video, k, k + "_app", tyname)
            
            if os.path.exists(rp_TC_clips):
                videos_list = os.listdir(rp_TC_clips)
                
                K_Coping_path = os.path.join(k_path, 'Coping')
                K_Training_path = os.path.join(k_path, 'Training')
                Coping_list, Training_list = [], []
                
                # Sort videos into respective categories, ignoring 'Uebung' files
                for vid in videos_list:
                    if "Training_1" in vid and 'Uebung' not in vid:
                        Coping_list.append(os.path.join(rp_TC_clips, vid))
                    elif "Training_2" in vid and 'Uebung' not in vid:
                        Training_list.append(os.path.join(rp_TC_clips, vid))
                        
                # Batch extract audio and remove silence
                if Coping_list:
                    v['data']['Coping'] = batch_extract_audio_without_silence(Coping_list, K_Coping_path)
                if Training_list:
                    v['data']['Training'] = batch_extract_audio_without_silence(Training_list, K_Training_path)
                    
    except Exception as e: 
        print(f"[{k}] Coping/Training Processing ERROR: {e}")

# ==========================================================
# 4. Save Final Dataset and Analyze Distributions
# ==========================================================
save_json(data_path, data)
print(f"\n[Success] Dataset saved to {data_path}")

# Calculate and print state distributions
coping_state = state_distribution(save_root, data, 'Coping')
ei_state = state_distribution(save_root, data, 'Ei')
training_state = state_distribution(save_root, data, 'Training')

print("\n--- Audio State Distributions ---")
print("Ei Distribution:", ei_state) 
print("Coping Distribution:", coping_state)
print("Training Distribution:", training_state)