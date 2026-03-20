# Audio-based Depression Detection (Audio-based-Depression-Detection-MLDA)

**Author:** Yi Zhang  
**Last Update:** 2026-03  

This project is an audio-based depression detection and emotion analysis system. It provides a complete pipeline ranging from video/audio extraction, denoising, speech transcription, and speaker diarization, all the way to depression classification and regression analysis using deep learning models (such as networks based on MFCC and Wav2Vec features).

---

## Project Structure

```text
.
├── Before_run.sh                       # Environment and task configuration script before running
├── analysis.ipynb                      # Interactive data analysis notebook for results and features
├── emotions_extract.py                 # Script for extracting emotion features
├── main_classification.py              # Main script for depression detection (classification task)
├── main_regression.py                  # Main script for depression severity (regression task)
├── configs/                            # Configuration directory
│   └── base_env.json                   # Base environment and global paths configuration
├── data/                               # Data and labels directory
│   ├── All_Emotion_Results.json        # Summary of all extracted emotion results
│   ├── extracted_full_dataset.json     # Extracted full feature dataset
│   ├── full_dataset.json               # Original full dataset configuration
│   ├── datasets/                       # Split datasets (e.g., Training and Coping)
│   ├── diff_files/                     # Differential data files directory
│   └── label/                          # Questionnaire scores and clinical diagnosis labels (CSV, XLSX, etc.)
├── Emotion_Analysis_Results/           # Directory for emotion analysis visualization charts
├── Output/                             # Model outputs and logs directory
│   ├── mfcc/                           # Cross-validation evaluation results for MFCC models
│   ├── wav2vec/                        # Cross-validation evaluation results for Wav2Vec models
│   └── outer_eval_log.json             # External evaluation log record
├── preprocess/                         # Core data preprocessing pipeline
│   ├── 01_label_collection.py          # Collect and clean clinical diagnosis/questionnaire labels
│   ├── 02_audiol_extract.py            # Extract audio segments from source video/audio files
│   ├── 03_transcription.py             # Speech transcription and speaker diarization (using Whisper, etc.)
│   ├── 04_extract_data_from_transcription_result.py # Extract structured data from transcription results
│   ├── 05_dataset_split.py             # Split datasets into training, validation, and testing sets
│   └── 06_add_diff_as_label.py         # Add differential features as training labels
├── src/                                # Core codebase for models and datasets
│   ├── datasets/                       # Dataset construction modules
│   │   ├── base_dataset.py             # Base dataset class
│   │   ├── builder.py                  # Dataset factory/builder
│   │   ├── mfcc_dataset.py             # Dataset class for MFCC audio features
│   │   └── wav2vec_dataset.py          # Dataset class for Wav2Vec audio features
│   ├── models/                         # Deep learning model modules
│   │   ├── builder.py                  # Model factory/builder
│   │   ├── mfcc_net.py                 # Neural network architecture based on MFCC features
│   │   └── wav2vec_net.py              # Neural network architecture based on Wav2Vec features
│   └── trainer.py                      # Unified controller for model training, validation, and testing
└── utils/                              # Utility functions directory
    ├── audio_process.py                # Low-level utilities for audio processing and denoising
    ├── label_process.py                # Utilities for label mapping and parsing
    └── utils.py                        # General miscellaneous utility functions
```
## Project Modules Summary
### 1. Data Preprocessing Pipeline (preprocess/)
This directory contains an automated data processing pipeline responsible for converting raw multimedia and questionnaire data into machine-readable formats for the models:

- Scripts 01-02: Handle label aggregation and basic audio segment extraction (uses FFmpeg to convert original videos into 44.1kHz stereo .wav files).

- Script 03: The core audio transcription and processing script. It applies Wiener denoising to audio channels, utilizes OpenAI's Whisper model to extract speech with word-level timestamps, and performs multi-speaker separation (Speaker Diarization).

- Scripts 04-06: Parse the generated JSON transcription files to extract useful text and acoustic segments, split the datasets, and build a label system with differential features for subsequent model training.

### 2. Deep Learning & Training Core (src/)
- datasets/: Encapsulates the data loading logic for different feature inputs. mfcc_dataset.py handles traditional Mel-Frequency Cepstral Coefficients (MFCC), while wav2vec_dataset.py interfaces with deep speech features extracted by modern pre-trained models.

- models/: Defines the deep neural network architectures used for depression classification and regression (mfcc_net.py and wav2vec_net.py).

- trainer.py: The main controller for the deep learning model's training lifecycle, including loss computation, backpropagation, and evaluation logic.

### 3. Main Entry Scripts (Root Directory)
- main_classification.py & main_regression.py: Top-level scripts used to launch the depression binary/multi-class classification task and the severity score regression prediction task, respectively.

- emotions_extract.py & analysis.ipynb: Used for Exploratory Data Analysis (EDA) to extract emotion feature distributions within the samples and generate related visualizations in the Emotion_Analysis_Results/ folder.

## How to Run
- Install Conda then explore enviroments
```bash
conda env create -f environment.yml
```
- [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) (Download and add to the system PATH for audio extraction)

- [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive) (for GPU-accelerated transcription)
  - After installation, update the `LD_LIBRARY_PATH` in `Before_run.sh` to point to your cuDNN installation directory.



## Running the Pipeline
### Step 1: Configure the Running Script
Modify and check the Before_run.sh file in the root directory:
If using GPU transcription, ensure the LD_LIBRARY_PATH is correctly pointing to your cuDNN installation directory.

Modify and check the configs.json file in the root directory:

Ensure the paths in `configs/base_env.json` are correctly configured for your environment:

* **`RAW_VIDEO_DIR`**: The absolute path to the directory containing the original raw multimedia (video/audio) files.
* **`EXTRACTED_AUDIO_DIR`**: The directory where the initially extracted, full-length audio files (e.g., converted `.wav` files) are stored.
* **`TRANSCRIPTION_DIR`**: The directory for saving the transcription and speaker diarization output files (e.g., generated `.json` files containing timestamps and text).
* **`FINAL_AUDIO_DIR`**: The directory where the final processed, separated, and segmented audio clips are saved for model training and evaluation.
* **`LOGS_DIR`**: The directory where the logs files are.(all of logs files)

### Step 2: Execute Data Preprocessing
Before running the models, you must execute the pipeline scripts in the preprocess/ directory sequentially (if the raw dataset has not been processed yet):
```bash
python preprocess/01_label_collection.py
python preprocess/02_audiol_extract.py
# ... execute sequentially up to 06
```

### Step 3: Model Training & Inference
Depending on your task requirements, run the corresponding main script directly. The system will read the relevant configurations from configs/base_env.json and call src/trainer.py for training and logging:

Classification Task (e.g., Distinguishing depressed/non-depressed):

```bash
python main_classification.py
```
Regression Task (e.g., Predicting continuous values like HRSD scores):

```bash
python main_regression.py
```

### Step 4: View Results

The cross-validation results and logs from the model runs will be saved in the Output/ directory (categorized into mfcc/ and wav2vec/ folders).

Visualizations and data analysis of emotions can be viewed by opening analysis.ipynb or by checking the files directly in the Emotion_Analysis_Results/ directory.