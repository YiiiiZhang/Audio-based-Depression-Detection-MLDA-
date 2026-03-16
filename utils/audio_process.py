import subprocess
import pandas as pd
import numpy as np
from pydub import AudioSegment
from collections import defaultdict
import subprocess
from pathlib import Path
from typing import Dict, List, Union
import re
import os
import json
from typing import Any, Tuple

def get_duration(wav_path: str) -> float:
    """
    Use ffprobe to obtain the duration of a WAV audio file (returned in seconds).
    """
    try:
        duration_seconds = float(
            subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                wav_path
            ]).decode().strip()
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to retrieve audio duration. Please verify that the path exists：{wav_path}")
    except ValueError:
        raise RuntimeError("Unable to parse ffprobe output; the audio file may be corrupted.")
    return duration_seconds

#============================================ EI SEGMENT EXTRACTION ============================================

def _sec_to_hms_ms(x):
    if x is None or pd.isna(x):
        return None
    ms = int(round((x - int(x)) * 1000))
    x = int(x)
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def _read_csv_safely(csv_path: str, encoding: str | None = None) -> pd.DataFrame:
    """
    Attempt to read with multiple encodings, using engine='python' to allow pandas 
    to auto-infer the separator (comma/semicolon/tab are all supported).
    """
    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            # sep=None + engine='python' lets pandas automatically identify the separator
            return pd.read_csv(csv_path, quotechar='"', encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("无法读取 CSV。")

def read_Ei_timerange(csv_path: str, encoding: str | None = None, start_index: int = 1):
    """
    Reads a CSV (column 1 = absolute timestamp, column 2 = label ei_xx, column 3 = text),
    extracts ei_01~ei_10 segments, and converts the time to be relative to the starting point;
    prefixes the ei label with the row number, e.g., 200_ei_01.

    Parameters
    ----
    csv_path : Path to the CSV file
    encoding : Optional, specifies the encoding; if not provided, it will try utf-8 / utf-8-sig / cp1252 / latin-1 sequentially
    start_index : Starting row number (default 1), used for the row number prefix

    Returns
    ----
    list[dict], where each dict contains:
      {
        "Time Range": ["HH:MM:SS.mmm", "HH:MM:SS.mmm" or None],
        "Text Content": <text from the third column>,
        "ei index": "<row_number>_ei_0x"
      }
    """
    df = _read_csv_safely(csv_path, encoding=encoding)

    if df.shape[1] < 3:
        raise ValueError("CSV 至少需要三列：时间、标签、文本。")

    # Only take the first three columns (some files might have empty trailing columns)
    df = df.iloc[:, :3].copy()

    # --- Parse and clean time (1st column) ---
    time_col = (
        df.iloc[:, 0].astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
        .str.replace(",", ".", regex=False)  # Compatible with comma decimals like 17:11:20,369723
    )
    ts = pd.to_datetime(time_col, errors="coerce")
    valid_mask = ts.notna()
    if not valid_mask.any():
        raise ValueError("时间列无法解析任何时间。")

    # Video start time = first "parsable" time
    t0 = ts[valid_mask].iloc[0]
    rel_sec = (ts - t0).dt.total_seconds()  # Invalid times become NaN

    # Labels and text
    labels = df.iloc[:, 1].astype(str).str.strip()
    texts  = df.iloc[:, 2].astype(str).fillna("").str.strip()

    # Select only ei_01 ~ ei_10
    ei_mask = labels.str.match(r"^ei_(0[1-9]|10)$")
    ei_idx  = np.flatnonzero(ei_mask.values)
    if len(ei_idx) == 0:
        return []

    # Prepare index table for the "next valid time"
    valid_idx = np.flatnonzero(valid_mask.values)

    def next_valid_after(i: int):
        pos = np.searchsorted(valid_idx, i + 1, side="left")
        return None if pos >= len(valid_idx) else valid_idx[pos]

    def prev_valid_upto(i: int):
        pos = np.searchsorted(valid_idx, i, side="right") - 1
        return None if pos < 0 else valid_idx[pos]

    tmp_results = defaultdict(list) #{01:[{ei_01:[]},{ei_01:[]}]}
    result = {"01":{},"02":{}}
    for i in ei_idx:
        # Start time: relative time of current row; if invalid, use the most recent valid time
        start_s = rel_sec.iloc[i]
        if pd.isna(start_s):
            pv = prev_valid_upto(i)
            if pv is None:
                # Skip if it is before the first valid time
                continue
            start_s = (ts.iloc[pv] - t0).total_seconds()

        # End time: next valid time; None if not found
        j = next_valid_after(i)
        end_s = None if j is None else (ts.iloc[j] - t0).total_seconds()

        tmp_results[labels.iloc[i].split('_')[-1]].append({labels.iloc[i] : [_sec_to_hms_ms(start_s), _sec_to_hms_ms(end_s)]}) 
    for key in tmp_results.keys():#{01:{ei_01:[],ei_02:[]}}
        result["01"] =result["01"] |  tmp_results[key][0]
        result["02"] =result["02"] |  tmp_results[key][1]
    return result

def extract_Ei_audio_segments(
    minx_key: str,
    value: Dict[str, Union[dict, list]],
    mp4_path: Union[str, Path],
    save_root: Union[str, Path],
    sr: int = 16000,
    channels: int = 1,
    audio_fmt: str = "wav",
) -> None:
    """
    Extract audio segments from an mp4 file based on the given app_logs time range.

    Parameters
    ----
    minx_key : Top-level key (e.g., "254"), used as the outermost directory name.
    value    : The value corresponding to this key; supports:
                {"01": {"ei_07": ["HH:MM:SS.mmm","HH:MM:SS.mmm"], ...}, "02": {...}}
    mp4_path : Path to the source mp4 file
    save_root: Root directory for saving outputs
    sr       : Sample rate (default 16k)
    channels : Number of audio channels (default 1)
    audio_fmt: Output audio format (default wav)
    """

    mp4_path = Path(mp4_path)
    save_root = Path(save_root)
    path_list=[]
    if not mp4_path.exists():
        raise FileNotFoundError(f"视频不存在: {mp4_path}")


    # Top-level directory: save_root/<minx_key>/
    base_dir = save_root / str(minx_key) / "Ei"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each sub-key (e.g., "01", "02"...)
    for sub_key, ei_map in value.items():
        if not isinstance(ei_map, dict):
            continue
        
        # Iterate through each ei segment 
        for ei_name, ts in ei_map.items():
            start_time,end_time = ts[0],ts[1]
            
            # Output filename: ei_07.wav
            out_name = f"{sub_key}_{ei_name}.{audio_fmt}"
            out_path = base_dir / out_name
            # ffmpeg command: millisecond precision; -vn removes video; export PCM/WAV
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", start_time,            # Start time
                "-to", end_time,              # End time (relative to input)
                "-i", str(mp4_path),        # Input file
                "-vn",                      # Remove video track
                "-acodec", "pcm_s16le" if audio_fmt.lower() == "wav" else "aac",
                "-ar", str(sr),             # Sample rate
                "-ac", str(channels),       # Audio channels
                "-y"
            ]

            # Encapsulation flags for different formats
            if audio_fmt.lower() == "wav":
                cmd += ["-f", "wav"]
            else:
                cmd += ["-n"]

            cmd += [str(out_path)]

            subprocess.run(cmd, check=True)
            path_list.append(out_name)
    return path_list


#============================================ Coping and Tranning SEGMENT EXTRACTION ============================================
mapping_dict = {
    'Aufgabe_1': 1,
    'Aufgabe_2': 2,
    'Aufgabe_3': 3,
    'Aufgabe_4': 4,
    'Aufgabe_5': 5,
    'Aufgabe_6': 6,
    'Aufgabe_7': 7,
    'Aufgabe_8': 8,
    'Aufgabe_9': 9,
    'Aufgabe_10': 10,
    'Aufgabe_11': 11,
    'Aufgabe_12': 12,
    'Aufgabe_13': 13,
    'Aufgabe_14': 14,
    'Aufgabe_15': 15,
    'Aufgabe_16': 16,
    'Aufgabe_17': 17,
    'Aufgabe_18': 18,
    'Aufgabe_19': 19,
    'Aufgabe_20': 20
}
def extract_audio_without_silence_single(mp4_path, save_wav_path,
                                         silence_thresh=-35,      # Silence threshold (dB)
                                         silence_duration=0.3):   # Silence duration threshold (seconds)
    """
    Extract audio from a single mp4, remove silence, and save as a 16kHz, mono WAV file.
    """

    # ----------------- 1. Extract entire audio to tmp first -----------------
    tmp_audio = save_wav_path.replace(".wav", "_tmp.wav")
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        tmp_audio
    ]
    subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ----------------- 2. Detect silence using silencedetect -----------------
    cmd_silence = [
        "ffmpeg",
        "-i", tmp_audio,
        "-af", f"silencedetect=noise={silence_thresh}dB:d={silence_duration}",
        "-f", "null",
        "-"
    ]
    result = subprocess.run(cmd_silence, stderr=subprocess.PIPE, text=True)
    stderr = result.stderr
    silence_starts = [float(x) for x in re.findall(r"silence_start: (\d+\.?\d*)", stderr)]
    silence_ends   = [float(x) for x in re.findall(r"silence_end: (\d+\.?\d*)", stderr)]

    # ----------------- 3. If no silence is found, use the entire audio directly -----------------
    if len(silence_starts) == 0:
        os.rename(tmp_audio, save_wav_path)
        return save_wav_path

    # ----------------- 4. Get total duration -----------------
    cmd_duration = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        tmp_audio
    ]
    total_duration = float(subprocess.check_output(cmd_duration).decode().strip())

    if len(silence_ends) < len(silence_starts):
        silence_ends.append(total_duration)

    # ----------------- 5. Calculate non-silent segment intervals -----------------
    non_silent_segments = []
    prev = 0.0
    for s, e in zip(silence_starts, silence_ends):
        if s > prev:
            non_silent_segments.append((prev, s))
        prev = e
    if prev < total_duration:
        non_silent_segments.append((prev, total_duration))

    # ----------------- 6. Trim and concatenate all non-silent segments -----------------
    concat_list_path = save_wav_path.replace(".wav", "_concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for i, (st, ed) in enumerate(non_silent_segments):
            seg = save_wav_path.replace(".wav", f"_seg{i}.wav")
            cmd_cut = [
                "ffmpeg", "-y",
                "-i", tmp_audio,
                "-ss", str(st),
                "-to", str(ed),
                "-ac", "1",
                "-ar", "16000",
                seg
            ]
            subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f.write(f"file '{seg}'\n")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-ac", "1",
        "-ar", "16000",
        save_wav_path
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ----------------- 7. Clean up temporary files -----------------
    os.remove(tmp_audio)
    os.remove(concat_list_path)
    for i in range(len(non_silent_segments)):
        seg = save_wav_path.replace(".wav", f"_seg{i}.wav")
        if os.path.exists(seg):
            os.remove(seg)

    return save_wav_path


def batch_extract_audio_without_silence(mp4_dirs, save_dir,
                                        silence_thresh=-35,
                                        silence_duration=0.3):
    """
    Batch process all mp4 files in a directory:
    - mp4_dirs: List of folders containing multiple .mp4 files
    - save_dir: Output wav folder path (created automatically if it doesn't exist)

    Returns:
        A dict: { mp4_filename: full_output_wav_path }
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for mp4_path in mp4_dirs:
        name= re.search(r"(Aufgabe)_\d+", mp4_path).group()
        save_wav_path = os.path.join(save_dir, str(mapping_dict[name])+".wav")

        out_path = extract_audio_without_silence_single(
            mp4_path,
            save_wav_path,
            silence_thresh=silence_thresh,
            silence_duration=silence_duration
        )
        results.append(out_path.split('/')[-1])

    return results


#============================================ transcription extraction ============================================

def extract_interviewee_segments(json_file_path: str, audio_file_path: str,output_path: str) -> Tuple[List[str], float]:
    """
    Extract and merge interviewee (speaker_id: 0) audio segments based on JSON transcription files.

    Args:
        json_file_path (str): Path to the JSON file containing transcription and timestamps.
        audio_file_path (str): Path to the original audio file (e.g., WAV or MP3).
        output_path (str): Path to save the resulting wav file.

    Returns:
        Tuple[List[str], float]: A tuple containing the list of output paths and the total duration.

    Raises:
        ValueError: If JSON file fails to load or interviewee ID is incorrect.
        FileNotFoundError: If the audio file is not found.
    """
    # 1. Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)
    interviewee_id = data['speaker_roles']['interviewee']
    # 3. Extract speech time intervals for the interviewee (ID 0)
    interviewee_intervals: List[Tuple[float, float]] = []
    for segment in data.get('segments', []):
        if segment.get('speaker_id') == interviewee_id:
            start_time = segment.get('start', 0.0)  # Start time in seconds
            end_time = segment.get('end', 0.0)      # End time in seconds
            if start_time < end_time:
                interviewee_intervals.append((start_time, end_time))
    # 4. Load original audio file
    original_audio = AudioSegment.from_file(audio_file_path)
    # 5. Cut and combine audio segments
    combined_audio = AudioSegment.empty()  # Initialize an empty AudioSegment
    for start_sec, end_sec in interviewee_intervals:
        # pydub uses milliseconds (ms), so conversion is needed
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        # Check if the end time exceeds the original audio length
        if end_ms > len(original_audio):
             end_ms = len(original_audio)
             if start_ms >= end_ms:
                 continue # Skip invalid or empty segments
        # Cut segment: [start_ms : end_ms]
        segment = original_audio[start_ms:end_ms]
        # Append segment to the combined audio
        combined_audio += segment
    # 6. Save the merged audio file
    if combined_audio.duration_seconds > 0:
        # Get format from output path (e.g., 'mp3', 'wav')
        output_format = output_path.split('.')[-1]
        combined_audio.export(output_path, format=output_format)

    return [output_path],combined_audio.duration_seconds

#============================================ Transcription Enrichment ============================================
def extract_interviewee_audio(data, audio_path: str, save_path: str) -> float:
    """
    Extract all audio segments of the interviewee based on the JSON structure, 
    merge them, save the file, and return the total duration.
    
    Args:
        data: JSON data containing speaker_roles and segments
        audio_path: Path to the original wav audio
        save_path: Path to save the result
        
    Returns:
        float: Total duration of the extracted audio in seconds
    """
    
    # 1. Load JSON data (handled upstream)
    
    # 2. Get the interviewee's ID
    # Structure example: "speaker_roles": {"psychologist": 1, "interviewee": 0}
    try:
        target_id = data['speaker_roles']['interviewee']
    except KeyError:
        print("Error: JSON中未找到 ['speaker_roles']['interviewee']")
        return 0.0

    # 3. Load original audio
    # pydub might consume memory for large files, but it is very convenient to use
    try:
        original_audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error: 无法加载音频文件 - {e}")
        return 0.0

    # 4. Extract segments
    # Initialize an empty AudioSegment for concatenation
    combined_audio = AudioSegment.empty()
    segments_found = 0
    
    for segment in data['segments']:
        if segment['speaker_id'] == target_id:
            # JSON time is usually in seconds, pydub needs milliseconds
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            # Slicing
            # This slicing operation is very similar to Python list slicing
            audio_chunk = original_audio[start_ms:end_ms]
            # Concatenation
            combined_audio += audio_chunk
            segments_found += 1
            
    # 5. Save results
    if len(combined_audio) > 0:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Export the file
        combined_audio.export(save_path, format="wav")
        # Calculate total duration (pydub len() returns milliseconds)
        duration_sec = len(combined_audio) / 1000.0
        return duration_sec
    else:
        return 0.0