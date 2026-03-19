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

# Import necessary utility functions
from utils.utils import read_json, save_json, state_distribution
from utils.audio_process import extract_interviewee_audio

def extract_and_save_transcriptions(data, trans_path, orig_audio_path, ext_audio_path):
    """
    Extracts the interviewee's audio based on transcription files, saves the 
    processed audio to a new directory, and updates the data dictionary.
    """
    # Create a shallow copy to safely update the dictionary
    extracted_data = data.copy()
    
    for k, v in tqdm(extracted_data.items(), desc="Extracting interviewee audios"):
        for label_type in ['Coping', 'Training']:
            # Skip if this user doesn't have the required audio type
            if label_type not in v.get('data', {}):
                continue
                
            wav_list = v['data'][label_type]
            
            # Setup destination folder for the extracted audio
            extracted_k = os.path.join(ext_audio_path, k, label_type)
            os.makedirs(extracted_k, exist_ok=True)
            
            wavs_dict = {}
            for wav in wav_list:
                # Resolve file paths
                transcription_file = os.path.join(trans_path, k, label_type, wav.replace('.wav', '.json'))
                original_file = os.path.join(orig_audio_path, k, label_type, wav)
                extracted_file = os.path.join(extracted_k, wav)
                
                # Ensure transcription exists before processing
                if not os.path.exists(transcription_file):
                    continue
                
                # Extract the interviewee's specific audio and calculate its duration
                duration = extract_interviewee_audio(
                    read_json(transcription_file), 
                    original_file, 
                    extracted_file
                )
                
                # Store valid extractions into the new dictionary structure
                if duration > 0.0:
                    wavs_dict[wav] = {'path': wav, 'duration': duration}
                    
            # Overwrite the original list with the new detailed dictionary
            v['data'][label_type] = wavs_dict
            
    return extracted_data


# ==========================================================
# Main Execution Block
# ==========================================================
if __name__ == "__main__":
    
    # 1. Load configuration and paths
    # Note: Using the configs/base_env.json path established in previous steps. 
    # If using root config, change to: os.path.join(project_root, 'configs.json')
    config_path = os.path.join(project_root, 'configs', 'base_env.json')
    BASE_CONFIG = read_json(config_path)
    
    transcription_path = BASE_CONFIG['TRANSCRIPTION_DIR']
    origin_audio_path = BASE_CONFIG['EXTRACTED_AUDIO_DIR']
    extracted_audio_path = BASE_CONFIG['FINAL_AUDIO_DIR']
    
    data_path = './data/full_dataset.json'
    extracted_datasets_path = './data/extracted_full_dataset.json'
    
    # Read the original full dataset
    data = read_json(data_path)
    
    # 2. Extract and save interviewee transcriptions
    extracted_data = extract_and_save_transcriptions(
        data=data,
        trans_path=transcription_path,
        orig_audio_path=origin_audio_path,
        ext_audio_path=extracted_audio_path
    )
    
    # Save the updated dataset
    save_json(extracted_datasets_path, extracted_data)
    print(f"\n[Success] Extracted dataset saved to {extracted_datasets_path}\n")
    
    # 3. Compute and print State Distributions
    # Calculate distributions directly from the memory variable (extracted_data)
    rp = extracted_audio_path
    
    coping_state = state_distribution(rp, extracted_data, 'Coping', interval_min=0.5)
    ei_state = state_distribution(rp, extracted_data, 'Ei', interval_min=0.5)
    training_state = state_distribution(rp, extracted_data, 'Training', interval_min=0.5)
    
    print("--- Audio State Distributions (0.5 min intervals) ---")
    print("Coping State Distribution:", coping_state)
    print("Ei State Distribution:", ei_state) 
    print("Training State Distribution:", training_state)