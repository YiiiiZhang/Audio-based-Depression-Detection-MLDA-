# Import necessary libraries
import os
import json
import argparse
import sys
import tempfile
import subprocess
from typing import Tuple, List, Dict, Optional

#Import the modules in the utils folder
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import torchaudio
import scipy.signal
from tqdm import tqdm
from faster_whisper import WhisperModel
from modelscope.pipelines import pipeline
import re
from collections import Counter

import time

from utils.utils import read_json, save_json
os.environ['MODELSCOPE_CACHE'] = '../Model/clearVoice'
def identify_speakers_by_keywords(enriched_result: List[Dict]) -> Dict[str, int]:
    # Corrected regex: using a single backslash to represent word boundaries
    keywords = [
        r'\bFragebogen\b', r'\bSkala\b', r'\bStimmung\b', r'\bMood\b',
        r'\bWie\b.*\?', r'\bHaben Sie\b', r'\bMöchten Sie\b',
        r'\bSie\b', r'\bIhnen\b', r'\bIhre\b'
    ]
    speaker_counters = Counter()
    ich_counter = Counter()

    for seg in enriched_result:
        spk = seg["speaker_id"]      # Numeric id, 0 or 1
        words = seg["words"]
        text = " ".join([w["word"] for w in words])
        keyword_count = sum(len(re.findall(kw, text, flags=re.IGNORECASE)) for kw in keywords)
        ich_count = len(re.findall(r'\bich\b', text, flags=re.IGNORECASE))
        speaker_counters[spk] += keyword_count
        ich_counter[spk] += ich_count

    if speaker_counters[0] != speaker_counters[1]:
        return {"psychologist": 0, "interviewee": 1} if speaker_counters[0] > speaker_counters[1] else {"psychologist": 1, "interviewee": 0}
    return {"psychologist": 1, "interviewee": 0} if ich_counter[0] > ich_counter[1] else {"psychologist": 0, "interviewee": 1}

def transcribe_minute_range_pipeline(
    mp4_path: str,
    minute_range: Optional[Tuple[int, int]] = None,
    diar_model_id: str = "iic/speech_campplus_speaker-diarization_common",
    diar_model_cache: str = "../Model/clearVoice",
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: Optional[str] = None,  # Allow automatic selection
    target_sr: int = 16000,
    eps: float = 1e-8,  # No longer used
) -> Dict:
    # Get duration
    try:
        total_duration = float(subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mp4_path
        ]).decode().strip())
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to get duration from: {mp4_path}")

    # Time cropping
    start_sec = 0
    if minute_range:
        start_min, end_min = minute_range
        start_sec = max(0, int(start_min) * 60)
        duration_sec = max(0.0, min((int(end_min) - int(start_min)) * 60, total_duration - start_sec))
    else:
        duration_sec = total_duration

    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        extracted_wav = tmp_wav.name
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", mp4_path,
        "-t", str(duration_sec), "-vn", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1", extracted_wav
    ]
    proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0 or not os.path.exists(extracted_wav):
        raise RuntimeError(f"ffmpeg failed to extract audio from: {mp4_path}")

    # Read and resample (keep on CPU)
    waveform, sr = torchaudio.load(extracted_wav)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Wiener denoising (remove added noise)
    cleaned = [torch.from_numpy(scipy.signal.wiener(waveform[ch].numpy())) for ch in range(waveform.shape[0])]
    waveform = torch.stack(cleaned).clamp(-1, 1).float()

    # Save temporary waveform
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    torchaudio.save(wav_path, waveform.cpu(), target_sr)

    # Whisper inference: automatic compute_type based on device
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    if compute_type is None:
        compute_type = "float16" if use_cuda else "int8"  # int8/int8_float32 are both fine on CPU
    model = WhisperModel(model_size, device=("cuda" if use_cuda else "cpu"), compute_type=compute_type)
    segments, _ = model.transcribe(wav_path, language="de", word_timestamps=True)
    word_items = [
        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
        for seg in segments if hasattr(seg, "words") and seg.words
        for w in seg.words
    ]

    # Speaker diarization: removed non-existent model_revision
    diar = pipeline(
        task="speaker-diarization",
        model=diar_model_id,
        model_cache_dir=diar_model_cache
    )
    diar_result = diar(wav_path, oracle_num=2)

    # Unified parsing response: supports list or dict, supports 'SPEAKER_00'/'00'/0
    def to_spk_id(x) -> int:
        if isinstance(x, int):
            return x
        s = str(x)
        m = re.search(r'(\d+)$', s)
        return int(m.group(1)) if m else 0

    if isinstance(diar_result, dict) and "text" in diar_result:
        diar_segs_raw = diar_result["text"]
    else:
        diar_segs_raw = diar_result

    diar_segments = []
    for seg in diar_segs_raw:
        if isinstance(seg, list) and len(seg) >= 3:
            start, end, spk = float(seg[0]), float(seg[1]), to_spk_id(seg[2])
        elif isinstance(seg, dict):
            start, end, spk = float(seg["start"]), float(seg["end"]), to_spk_id(seg["speaker"])
        else:
            continue
        diar_segments.append({"start": start, "end": end, "speaker_id": spk, "speaker_label": f"SPEAKER_{spk:02d}"})

    # Align word-level timestamps
    enriched_result = []
    for ds in diar_segments:
        start, end, spk = ds["start"], ds["end"], ds["speaker_id"]
        words = [w for w in word_items if (w["start"] is not None) and (start <= w["start"] < end)]
        enriched_result.append({
            "speaker_id": spk,
            "speaker_label": f"SPEAKER_{spk:02d}",
            "start": start,
            "end": end,
            "words": words
        })

    # Cleanup
    try:
        os.remove(wav_path)
    finally:
        try:
            os.remove(extracted_wav)
        except Exception:
            pass

    # Role identification
    roles = identify_speakers_by_keywords(enriched_result)
    return {"speaker_roles": roles, "segments": enriched_result}

def batch_transcription(data_rp, save_rp, data_list):
    path_list = []
    for file_path in data_list:
        wav_path = os.path.join(data_rp, file_path)
        save_path = os.path.join(save_rp, file_path.replace('.wav', '.json'))
        if os.path.exists(save_path):
            path_list.append(save_path)
            continue
        enriched_result = transcribe_minute_range_pipeline(wav_path)
        save_json(save_path, enriched_result)
        path_list.append(save_path)
    return path_list


if __name__ == "__main__":
    label_type_list = ['Coping','Training'] 
    Base_path = read_json(os.path.join(project_root, 'configs', 'base_env.json'))
    DATA_DIR = Base_path['EXTRACTED_AUDIO_DIR']
    SAVE_DIR = Base_path['TRANSCRIPTION_DIR']
    for label_type in label_type_list:
        ERROR_JSON = f"./data/{label_type}_Error.json"
        Finished_JSON = f"./data/{label_type}_Finished.json"

        os.makedirs(SAVE_DIR, exist_ok=True)
        error_dict = {}
        finished_dict = {}

        data = read_json("./data/full_dataset.json")
        user_list = list(data.keys())
        user_list.sort()
        user_list = user_list[len(user_list)//2:]
        for k in user_list:
            id_p = os.path.join(DATA_DIR, k)
            id_s = os.path.join(SAVE_DIR, k)
            os.makedirs(id_s, exist_ok=True)
            v = data[k]
            if label_type not in v['data']:
                continue
            id_data_path = os.path.join(id_p, label_type) #base/k/label_type
            id_save_path = os.path.join(id_s, label_type)
            os.makedirs(id_save_path, exist_ok=True)
            try:
                singl_id_result = batch_transcription(id_data_path, id_save_path, v['data'][label_type]) #{'Coping':[xxx.wav,...]}
                finished_dict[k] = singl_id_result
            except Exception as e:
                error_dict[k] = str(e)
        save_json(ERROR_JSON, error_dict)
        save_json(Finished_JSON, finished_dict)