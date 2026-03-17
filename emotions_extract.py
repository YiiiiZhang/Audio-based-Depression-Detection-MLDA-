import os
import sys
import torch
from tqdm import tqdm
from speechbrain.inference.interfaces import foreign_class

# ================= Path Setting =================
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)

# Add the project root to the Python search path so we can find 'utils'
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.utils import read_json, save_json

# ================= Configuration =================
# Define the category labels to be recognized
TARGET_LABELS = ['Coping', 'Training']
# Output filename
OUTPUT_FILENAME = "../data/All_Emotion_Results.json"
# =================================================

# 1. Load Model (Automatically selects GPU/CPU)
run_opts = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
print(f"Loading Emotion Recognition Model on {run_opts['device']}...")

# Added `savedir` to route the wav2vec model checkpoints to the specified directory
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
    pymodule_file="custom_interface.py", 
    classname="CustomEncoderWav2vec2Classifier",
    savedir="./Model/wav2vec2_checkpoints", 
    run_opts=run_opts
)
print("Model loaded successfully.")

def predict_emotion(wav_path):
    """
    Performs inference on a single wav file and returns the result in a formatted dictionary.
    """
    try:
        # classify_file returns: out_prob, score, index, text_lab
        out_prob, score, index, text_lab = classifier.classify_file(wav_path)
        
        return {
            "emotion": text_lab[0],               # Emotion label (str)
            "out_prob": out_prob[0].cpu().tolist() # Probability distribution (list)
        }
    except Exception as e:
        print(f"Error predicting {wav_path}: {e}")
        return None

def main():
    # Read path configuration
    # Note: Assumes configs.json is in the project root
    base_path = read_json("../data/configs.json")
    DATA_DIR = base_path['FINAL_AUDIO_DIR']  # Audio root directory
    
    # Read the full dataset index
    full_data = read_json("../data/extracted_full_dataset.json")
    
    # Used to store the final comprehensive JSON result
    # Structure: {user_id: {label_type: [result_dict, ...]}}
    all_results = {}

    # Iterate through all users (User ID)
    # Using tqdm to show overall progress
    for user_id, user_content in tqdm(full_data.items(), desc="Processing Users"):
        
        user_results = {} # Stores results for the current user
        user_audio_root = os.path.join(DATA_DIR, user_id)
        
        # Iterate through each category that needs processing (e.g., Coping, Training)
        for label_type in TARGET_LABELS:
            
            # Check if the user has data for this specific category
            if 'data' in user_content and label_type in user_content['data']:
                file_list = user_content['data'][label_type]
                
                # Corresponding audio folder path
                audio_dir = os.path.join(user_audio_root, label_type)
                
                # Store all recognition results under this category
                label_results_list = []
                
                # Iterate through all audio files under this category
                # Note: file_list usually contains filenames like ['1.wav', '2.wav']
                for file_name in file_list:
                    wav_path = os.path.join(audio_dir, file_name)
                    
                    if os.path.exists(wav_path):
                        # Execute prediction
                        res = predict_emotion(wav_path)
                        if res:
                            label_results_list.append({file_name: res})
                    else:
                        # If the file does not exist, we skip it.
                        # Alternatively, you could append an error marker like {"error": "file_missing"}
                        print(f"Warning: File missing {wav_path}")

                # Save this category's results to the user's dictionary
                if label_results_list:
                    user_results[label_type] = label_results_list
        
        # Save all results for the current user to the master dictionary
        if user_results:
            all_results[user_id] = user_results

    # After the loop finishes, save all results at once
    print(f"Saving all results to {OUTPUT_FILENAME}...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    save_json(OUTPUT_FILENAME, all_results)

if __name__ == "__main__":
    main()