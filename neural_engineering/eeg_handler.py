import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from datetime import datetime

# --- [그래프 설정] ---
import matplotlib
matplotlib.use('Agg') # 화면 표시 없이 파일 저장 전용 모드
import matplotlib.pyplot as plt

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

def get_latest_rawdata_path(base_path="C:/MAVE_RawData"):
    # 1순위: C드라이브 경로 확인
    if os.path.exists(base_path):
        try:
            all_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if all_folders:
                latest_folder_path = max(all_folders, key=os.path.getctime)
                target_file = os.path.join(latest_folder_path, "Rawdata.txt")
                if os.path.exists(target_file): return target_file
        except: pass
    
    # 2순위: 프로젝트 폴더 내 파일 확인
    local_file = "Rawdata.txt"
    if os.path.exists(local_file):
        return os.path.abspath(local_file)
    return None

def calculate_concentration_index(processor, raw_signal):
    epsilon = 1e-10
    theta_wave = processor.butter_bandpass_filter(raw_signal, 4.0, 8.0)
    alpha_wave = processor.butter_bandpass_filter(raw_signal, 8.0, 13.0)
    beta_wave = processor.butter_bandpass_filter(raw_signal, 13.0, 30.0)
    
    theta_power = processor.moving_average(processor.get_power(theta_wave))
    alpha_power = processor.moving_average(processor.get_power(alpha_wave))
    beta_power = processor.moving_average(processor.get_power(beta_wave))
    
    ba_ratio = beta_power / (alpha_power + epsilon)
    bt_ratio = beta_power / (theta_power + epsilon)
    return (ba_ratio + bt_ratio) / 2.0

def save_analysis_log(log_lines):
    try:
        with open("analysis_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
    except: pass

# ✅ 수정됨: 전체 데이터를 받아서 Baseline 구분선을 그어주는 함수
def save_z_score_plot(full_z_fp1, full_z_fp2, threshold, ceiling, baseline_sec, fs=256):
    try:
        plt.close('all')
        
        # 전체 시간축 생성
        total_seconds = len(full_z_fp1) / fs
        time_axis = np.linspace(0, total_seconds, len(full_z_fp1))
        
        plt.figure(figsize=(10, 5))
        
        # 전체 데이터 그리기
        plt.plot(time_axis, full_z_fp1, label='Fp1 Z-Score', color='blue', alpha=0.6, linewidth=1)
        plt.plot(time_axis, full_z_fp2, label='Fp2 Z-Score', color='orange', alpha=0.6, linewidth=1)
        
        # 기준선 그리기
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Concentration Threshold ({threshold})')
        
        # ✅ Baseline(30초) 구분선 추가 (빨간 점선)
        plt.axvline(x=baseline_sec, color='red', linestyle=':', linewidth=2, label='End of Baseline (30s)')
        
        # 그래프 꾸미기
        plt.title(f"Full Z-Score Flow (Total: {total_seconds:.1f}s)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Z-Score (rel. to Baseline)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=-2, top=min(ceiling + 1, 10)) # y축 범위 안정화

        # 저장 (절대 경로)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        save_path = os.path.join(project_dir, "z_score_graph.png")
        
        plt.savefig(save_path)
        print(f"📊 그래프 저장 완료: {save_path}")
        plt.close()
    except Exception as e:
        print(f"🚨 그래프 저장 실패: {e}")

def analyze_concentration_intervals(file_path, baseline_sec=30.0, z_threshold=0.7, z_ceiling=4.0):
    log_buffer = []
    print(f"🔍 분석 시작: {file_path}")
    
    try:
        try:
            df = pd.read_csv(file_path, delimiter="\t", encoding='cp949')
        except:
            df = pd.read_csv(file_path, delimiter="\t", encoding='utf-8')

        if df.empty: return []

        fs = 256
        processor = SignalProcessor(fs)
        
        col_fp1 = next((c for c in df.columns if 'Fp1' in c), df.columns[1])
        col_fp2 = next((c for c in df.columns if 'Fp2' in c), df.columns[2])

        signal_fp1 = df[col_fp1].values
        signal_fp2 = df[col_fp2].values
        
        # 전체 길이에 대한 지표 계산
        idx_fp1 = calculate_concentration_index(processor, signal_fp1)
        idx_fp2 = calculate_concentration_index(processor, signal_fp2)
        
        base_samples = int(baseline_sec * fs)
        
        # 데이터가 너무 짧아도 일단 처리는 시도
        if len(idx_fp1) <= base_samples:
            print(f"⚠️ 경고: 데이터 길이({len(idx_fp1)/fs:.1f}초)가 Baseline({baseline_sec}초)보다 짧습니다.")
            # 짧아도 에러 안 나게 강제 설정
            base_fp1 = idx_fp1
            base_fp2 = idx_fp2
        else:
            base_fp1 = idx_fp1[:base_samples]
            base_fp2 = idx_fp2[:base_samples]
        
        # Z-Score 변환
        std_fp1 = np.std(base_fp1) if np.std(base_fp1) > 1e-10 else 1.0
        std_fp2 = np.std(base_fp2) if np.std(base_fp2) > 1e-10 else 1.0
        
        z_fp1 = (idx_fp1 - np.mean(base_fp1)) / std_fp1
        z_fp2 = (idx_fp2 - np.mean(base_fp2)) / std_fp2
        
        # ✅ [핵심 수정 1] 분석 결과와 상관없이 전체 그래프를 무조건 그림
        # (잘린 데이터가 아닌 'z_fp1' 전체를 넘김)
        save_z_score_plot(z_fp1, z_fp2, z_threshold, z_ceiling, baseline_sec, fs)

        # 실제 분석 (30초 이후부터)
        if len(z_fp1) > base_samples:
            task_z_fp1 = z_fp1[base_samples:]
            task_z_fp2 = z_fp2[base_samples:]
        else:
            print("🛑 Baseline 이후 데이터가 없어 구간 분석을 종료합니다.")
            return []

        # 구간 검출 로직
        is_active = ((task_z_fp1 > z_threshold) | (task_z_fp2 > z_threshold))
        
        intervals = []
        start = None
        
        # i는 task 시작 시점(0초) 기준임. 나중에 baseline_sec를 더해줘야 실제 시간.
        for i, active in enumerate(is_active):
            curr_task_time = i / fs
            if active and start is None:
                start = curr_task_time
            elif not active and start is not None:
                if curr_task_time - start >= 3.0: 
                    # 실제 영상 시간 = task 시간 + 30초
                    intervals.append((start + baseline_sec, curr_task_time + baseline_sec))
                start = None
                
        if start is not None:
            end_task_time = len(is_active)/fs
            if end_task_time - start >= 3.0:
                 intervals.append((start + baseline_sec, end_task_time + baseline_sec))

        # ✅ [핵심 수정 2] 구간이 없어도 에러 메시지 대신 로그만 남기고 종료
        if not intervals:
            save_analysis_log(["집중 구간 없음 (그래프 확인 요망)"])
            print("💡 집중 구간이 발견되지 않았습니다. 그래프를 확인하세요.")
            # 테스트를 위해 전체 길이의 일부를 강제로 반환하고 싶다면 아래 주석 해제
            # total_len = len(z_fp1)/fs
            # return [(total_len*0.4, total_len*0.6)] 
            return []

        return intervals

    except Exception as e:
        print(f"🚨 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_timestamp_durations_from_file(file_path=None, ignored=None):
    # 수정된 로직: 인자로 받은 file_path가 있으면 그것을 우선 사용
    if file_path and os.path.exists(file_path):
        target_path = file_path
    else:
        # 없으면 기존처럼 자동 탐색
        target_path = get_latest_rawdata_path()
    
    # 여기서 기준값(Threshold)을 조절하세요 (현재 0.7)
    FIXED_THRESHOLD = 0.7 

    if target_path:
        print(f"📂 파일 로드 및 분석 시작: {target_path}")
        # analyze_concentration_intervals 함수는 전체 경로(파일명 포함)를 필요로 함
        return analyze_concentration_intervals(target_path, z_threshold=FIXED_THRESHOLD)
    else:
        print("🚨 유효한 Rawdata.txt 파일을 찾을 수 없습니다.")
        return []