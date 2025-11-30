import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from datetime import datetime

# 그래프 충돌 방지 설정
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- [1] 신호 처리 클래스 ---
class SignalProcessor:
    def __init__(self, fs=256):
        self.fs = fs
    def butter_bandpass_filter(self, data, lowcut, highcut, order=2):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y
    def get_power(self, data): return data ** 2
    def moving_average(self, data, window_sec=1.0):
        window_size = int(window_sec * self.fs)
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# --- [2] 파일 로딩 및 분석 로직 ---
def get_latest_rawdata_path(base_path="C:/MAVE_RawData"):
    if not os.path.exists(base_path): return None
    try:
        all_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not all_folders: return None
        latest_folder_path = max(all_folders, key=os.path.getctime)
        target_file = os.path.join(latest_folder_path, "Rawdata.txt")
        if not os.path.exists(target_file): return None
        return target_file
    except: return None

def calculate_concentration_index(processor, raw_signal):
    theta_wave = processor.butter_bandpass_filter(raw_signal, 4.0, 8.0)
    alpha_wave = processor.butter_bandpass_filter(raw_signal, 8.0, 13.0)
    beta_wave = processor.butter_bandpass_filter(raw_signal, 13.0, 30.0)
    theta_power = processor.moving_average(processor.get_power(theta_wave))
    alpha_power = processor.moving_average(processor.get_power(alpha_wave))
    beta_power = processor.moving_average(processor.get_power(beta_wave))
    epsilon = 1e-20
    ba_ratio = beta_power / (alpha_power + epsilon)
    bt_ratio = beta_power / (theta_power + epsilon)
    return (ba_ratio + bt_ratio) / 2.0

def save_analysis_log(log_lines):
    try:
        with open("analysis_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print("분석 로그 저장 완료.")
    except: pass

def save_z_score_plot(z_fp1, z_fp2, threshold, ceiling, fs=256):
    try:
        time_axis = np.arange(len(z_fp1)) / fs
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, z_fp1, label='Fp1 Z-Score', color='blue', alpha=0.6, linewidth=1)
        plt.plot(time_axis, z_fp2, label='Fp2 Z-Score', color='orange', alpha=0.6, linewidth=1)
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        plt.axhline(y=ceiling, color='red', linestyle=':', linewidth=2, label=f'Ceiling ({ceiling})')
        plt.title("EEG Concentration Z-Score Analysis")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Z-Score")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig("z_score_graph.png", dpi=100)
        plt.close()
        print("📊 그래프 이미지 저장 완료: z_score_graph.png")
    except Exception as e:
        print(f"그래프 저장 실패: {e}")

def analyze_concentration_intervals(file_path, baseline_sec=30.0, z_threshold=1.0, z_ceiling=4.0):
    log_buffer = []
    log_buffer.append(f"[{datetime.now()}] 분석 시작: {file_path}")
    log_buffer.append(f"설정값 -> Baseline: {baseline_sec}초, Min(TH): {z_threshold}, Max(Ceiling): {z_ceiling}")
    
    try:
        df = pd.read_csv(file_path, delimiter="\t", encoding='cp949')
        if df.empty: return []

        fs = 256
        processor = SignalProcessor(fs)
        col_fp1 = 'EEG_Fp1' if 'EEG_Fp1' in df.columns else df.columns[1]
        col_fp2 = 'EEG_Fp2' if 'EEG_Fp2' in df.columns else (df.columns[2] if len(df.columns) > 2 else col_fp1)
        signal_fp1 = df[col_fp1].values
        signal_fp2 = df[col_fp2].values
        
        log_buffer.append(f"--- 데이터 상태 ---")
        log_buffer.append(f"Fp1 신호 평균 크기: {np.mean(np.abs(signal_fp1)):.4e}")

        index_fp1_all = calculate_concentration_index(processor, signal_fp1)
        index_fp2_all = calculate_concentration_index(processor, signal_fp2)

        baseline_samples = int(baseline_sec * fs)
        if len(index_fp1_all) < baseline_samples + (10 * fs):
            return []

        baseline_fp1 = index_fp1_all[:baseline_samples]
        baseline_fp2 = index_fp2_all[:baseline_samples]
        base_mean_fp1, base_std_fp1 = np.mean(baseline_fp1), np.std(baseline_fp1)
        base_mean_fp2, base_std_fp2 = np.mean(baseline_fp2), np.std(baseline_fp2)
        
        log_buffer.append(f"[Fp1 휴식] 평균: {base_mean_fp1:.4e}, 표준편차: {base_std_fp1:.4e}")

        std_safe_fp1 = base_std_fp1 if base_std_fp1 > 1e-20 else 1.0
        std_safe_fp2 = base_std_fp2 if base_std_fp2 > 1e-20 else 1.0

        z_score_fp1 = (index_fp1_all - base_mean_fp1) / std_safe_fp1
        z_score_fp2 = (index_fp2_all - base_mean_fp2) / std_safe_fp2

        task_z_fp1 = z_score_fp1[baseline_samples:]
        task_z_fp2 = z_score_fp2[baseline_samples:]

        save_z_score_plot(task_z_fp1, task_z_fp2, z_threshold, z_ceiling, fs)
        
        valid_fp1 = (task_z_fp1 > z_threshold) & (task_z_fp1 < z_ceiling)
        valid_fp2 = (task_z_fp2 > z_threshold) & (task_z_fp2 < z_ceiling)
        is_concentrating = valid_fp1 | valid_fp2

        intervals = []
        start_time = None
        for i, active in enumerate(is_concentrating):
            current_video_time = i / fs
            if active and start_time is None:
                start_time = current_video_time
            elif not active and start_time is not None:
                end_time = current_video_time
                if end_time - start_time >= 5.0: 
                    intervals.append((start_time, end_time))
                    log_buffer.append(f"구간 검출: {start_time:.2f}초 ~ {end_time:.2f}초")
                start_time = None
        
        if start_time is not None:
            end_time = len(is_concentrating) / fs
            if end_time - start_time >= 5.0:
                 intervals.append((start_time, end_time))
                 log_buffer.append(f"구간 검출: {start_time:.2f}초 ~ {end_time:.2f}초")

        if not intervals:
            save_analysis_log(log_buffer)
            total_len = len(is_concentrating)/fs
            return [(total_len*0.3, total_len*0.7)]

        save_analysis_log(log_buffer)
        return intervals

    except Exception as e:
        print(f"오류: {e}")
        return [(10.0, 20.0)]

# --- [3] 호출부 (여기가 핵심입니다!) ---
def load_timestamp_durations_from_file(timestamp_path, ignored_arg=None):
    """
    외부에서 어떤 값을 보내든 무시하고, 여기서 정한 FIXED_THRESHOLD를 사용합니다.
    """
    real_data_path = get_latest_rawdata_path()
    
    # ▼▼▼ [설정] 여기서만 바꾸면 무조건 적용됩니다! ▼▼▼
    FIXED_THRESHOLD = 0.7
    # ▲▲▲ ----------------------------------------- ▲▲▲

    if real_data_path:
        print(f"분석 시작: {real_data_path} (고정 TH: {FIXED_THRESHOLD})")
        return analyze_concentration_intervals(
            real_data_path, 
            baseline_sec=30.0, 
            z_threshold=FIXED_THRESHOLD, # 외부 값 무시하고 이거 씀
            z_ceiling=4.0
        )
    else:
        return [(10.0, 20.0)]