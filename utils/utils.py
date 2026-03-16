import json
import re
import os
import shutil

def read_json(json_file):
    return json.load(open(json_file,'r',encoding='utf-8'))

def save_json(json_file,data):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def delete_contents(folder_path):
    """
    Completely delete a folder, including the folder itself and all its contents.
    """
    # 1. Check if the path exists to avoid errors
    if not os.path.exists(folder_path):
        print(f"[Warning] 路径不存在，跳过: {folder_path}")
        return

    try:
        # 2. Recursively delete the entire directory tree
        shutil.rmtree(folder_path)
        print(f"[Success] 已彻底删除: {folder_path}")
    except OSError as e:
        print(f"[Error] 删除失败 {folder_path}: {e}")

import os
import wave
from collections import defaultdict
def get_wav_duration_minutes(file_path):
    if not os.path.exists(file_path):
        return 0.0
    try:
        with wave.open(file_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration_seconds = frames / float(rate)
            return duration_seconds / 60.0
    except Exception as e:
        print(f"[Error] 读取 wav 失败 {file_path}: {e}")
        return 0.0
def analyze_audio_distribution(rp, file_list, interval_min=2.0):
    """
    args:
        interval_min (float): The length of the statistics interval in minutes, default is 2 minutes
    """
    # Use defaultdict for convenient counting
    stats_counter = defaultdict(int)
    total_files = 0
    total_duration_all = 0.0

    # Iterate through all lists in the dictionary
    for file_path in file_list:
        wav_path = os.path.join(rp, file_path)
        
        # 1. Call the previous function to get duration (assuming it returns minutes)
        duration = get_wav_duration_minutes(wav_path)
        
        # Accumulate total data
        total_files += 1
        total_duration_all += duration
        
        # 2. Calculate the interval (core logic update)
        # Logic: (current_duration // interval_length) * interval_length = interval_start
        # e.g.: duration 3.5, interval 2 -> 3.5 // 2 = 1.0 -> 1.0 * 2 = 2 (start)
        # e.g.: duration 0.8, interval 0.5 -> 0.8 // 0.5 = 1.0 -> 1.0 * 0.5 = 0.5 (start)
        start_interval = (duration // interval_min) * interval_min
        end_interval = start_interval + interval_min
        
        # 3. Format Label (to make the key look better)
        # If the interval is an integer (like 2.0), convert to int; if float (like 0.5), keep 1 decimal
        if float(interval_min).is_integer():
            s_str = int(start_interval)
            e_str = int(end_interval)
        else:
            s_str = round(start_interval, 2)
            e_str = round(end_interval, 2)

        range_label = f"{s_str}-{e_str} min"
        stats_counter[range_label] += 1

    return stats_counter

def state_distribution(rp,data,label_key,interval_min=2.0):
    distribution = defaultdict(int)
    for k,v in data.items():
        if label_key not in v['data']:
            continue
        id_path = os.path.join(rp,k,label_key)
        result = analyze_audio_distribution(id_path,v['data'][label_key],interval_min=interval_min) #{'01':[xxx.wav,...],...}
        for range_label, count in result.items():
            distribution[range_label] += count
    return dict(distribution)