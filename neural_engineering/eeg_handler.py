import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # 화면 표시 없이 파일 저장 전용 모드
import matplotlib.pyplot as plt

# ==================================================================================
# 1. 공통 유틸리티 (그래프 저장 등)
# ==================================================================================
def save_z_score_plot(time_axis, full_z_fp1, full_z_fp2, threshold, baseline_sec, mode="Rawdata"):
    try:
        plt.close('all')
        plt.figure(figsize=(10, 5))
        
        # 전체 데이터 그리기
        plt.plot(time_axis, full_z_fp1, label='Fp1 Z-Score', color='blue', alpha=0.6, linewidth=1)
        plt.plot(time_axis, full_z_fp2, label='Fp2 Z-Score', color='orange', alpha=0.6, linewidth=1)
        
        # 기준선 그리기
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Concentration Threshold ({threshold})')
        
        # Baseline(30초) 구분선
        plt.axvline(x=baseline_sec, color='red', linestyle=':', linewidth=2, label='End of Baseline (30s)')
        
        plt.title(f"Concentration Z-Score Flow ({mode} Mode)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Z-Score (rel. to Baseline)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=-2, top=5) # y축 범위 안정화
        # 저장 (절대 경로)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        save_path = os.path.join(project_dir, "z_score_graph.png")
        
        plt.savefig(save_path)
        print(f"📊 그래프 저장 완료: {save_path}")
        plt.close()
    except Exception as e:
        print(f"🚨 그래프 저장 실패: {e}")

def save_analysis_log(log_lines):
    try:
        with open("analysis_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
    except: pass

# ==================================================================================
# 2. [신규 기능] Biomarkers.txt 전용 처리기 (데이터 유실 시 강력 추천)
# ==================================================================================
def parse_time(time_str):
    """ 시간 문자열(19:06:22.019)을 datetime 객체로 변환 """
    try:
        return datetime.strptime(time_str.strip(), "%H:%M:%S.%f")
    except:
        # 포맷이 다를 경우 예외처리 (HH:MM:SS 등)
        try:
            return datetime.strptime(time_str.strip(), "%H:%M:%S")
        except:
            return datetime.now()

def analyze_biomarkers(file_path, video_duration_sec=196, baseline_sec=30.0, z_threshold=0.7):
    """
    Biomarkers.txt의 정확한 지표를 사용하여 데이터를 보간(Interpolation)하고 분석합니다.
    데이터 개수가 적어도(55개 등) 값 자체가 정확하므로 Rawdata보다 신뢰도가 높습니다.
    """
    print(f"✨ [Biomarkers 모드] 정밀 지표 데이터를 분석합니다: {file_path}")
    
    try:
        # 파일 읽기 (인코딩 대응)
        try:
            df = pd.read_csv(file_path, delimiter="\t", encoding='cp949')
        except:
            df = pd.read_csv(file_path, delimiter="\t", encoding='utf-8')
        # 컬럼 공백 제거
        df.columns = [c.strip() for c in df.columns]
        # 필수 컬럼 확인
        required_cols = ['Time', 'Fp1_Theta(%)', 'Fp1_Alpha(%)', 'Fp1_Beta(%)', 
                                 'Fp2_Theta(%)', 'Fp2_Alpha(%)', 'Fp2_Beta(%)']
        
        if not all(col in df.columns for col in required_cols):
            print("🚨 Biomarkers.txt 형식이 올바르지 않습니다.")
            return None
        # 1. 시간 축 생성 (0초 ~ 종료초)
        times = [parse_time(t) for t in df['Time']]
        start_time = times[0]
        # 각 데이터가 시작 후 몇 초 시점인지 계산 (예: 0.0s, 3.5s, 7.2s ...)
        original_seconds = np.array([(t - start_time).total_seconds() for t in times])
        # 2. 집중도 지표(Ratio) 계산
        # 공식: (Beta/Alpha + Beta/Theta) / 2
        epsilon = 1e-6 # 0 나누기 방지
        
        # Fp1 계산
        fp1_b = df['Fp1_Beta(%)'].values
        fp1_a = df['Fp1_Alpha(%)'].values + epsilon
        fp1_t = df['Fp1_Theta(%)'].values + epsilon
        idx_fp1 = ((fp1_b / fp1_a) + (fp1_b / fp1_t)) / 2.0
        
        # Fp2 계산
        fp2_b = df['Fp2_Beta(%)'].values
        fp2_a = df['Fp2_Alpha(%)'].values + epsilon
        fp2_t = df['Fp2_Theta(%)'].values + epsilon
        idx_fp2 = ((fp2_b / fp2_a) + (fp2_b / fp2_t)) / 2.0
        # 3. 데이터 보간 (Interpolation) -> 핵심 로직!
        # 55개의 점을 1960개(0.1초 간격)의 점으로 부드럽게 연결합니다.
        target_fs = 10 # 10Hz (0.1초 단위)
        target_len = int(video_duration_sec * target_fs)
        target_time_axis = np.linspace(0, video_duration_sec, target_len)
        
        interp_fp1 = np.interp(target_time_axis, original_seconds, idx_fp1)
        interp_fp2 = np.interp(target_time_axis, original_seconds, idx_fp2)
        # 4. Baseline 통계 산출 (앞 30초)
        base_mask = target_time_axis <= baseline_sec
        base_fp1 = interp_fp1[base_mask]
        base_fp2 = interp_fp2[base_mask]
        
        # 표준편차가 0이면 1로 설정 (에러 방지)
        mean_1, std_1 = np.mean(base_fp1), (np.std(base_fp1) if np.std(base_fp1) > 1e-6 else 1.0)
        mean_2, std_2 = np.mean(base_fp2), (np.std(base_fp2) if np.std(base_fp2) > 1e-6 else 1.0)
        # 5. Z-Score 변환
        z_fp1 = (interp_fp1 - mean_1) / std_1
        z_fp2 = (interp_fp2 - mean_2) / std_2
        # 6. 그래프 저장
        save_z_score_plot(target_time_axis, z_fp1, z_fp2, z_threshold, baseline_sec, mode="Biomarkers")
        # 7. 구간 추출 로직
        intervals = []
        start = None
        is_active = (z_fp1 > z_threshold) | (z_fp2 > z_threshold)
        for i, active in enumerate(is_active):
            curr_t = target_time_axis[i]
            
            # Baseline 구간(30초)은 무시
            if curr_t < baseline_sec: continue
            if active and start is None:
                start = curr_t
            elif not active and start is not None:
                if curr_t - start >= 3.0: # 3초 이상 지속 시 인정
                    intervals.append((start, curr_t))
                start = None
        
        # 마지막 구간 처리
        if start is not None and (target_time_axis[-1] - start >= 3.0):
            intervals.append((start, target_time_axis[-1]))
        if not intervals:
            save_analysis_log(["집중 구간 없음 (Biomarker 그래프 확인 요망)"])
            print("💡 집중 구간이 발견되지 않았습니다.")
            return []
        return intervals
    except Exception as e:
        print(f"🚨 Biomarkers 분석 중 오류: {e}")
        return None # None을 반환하면 Rawdata 모드로 넘어감

# ==================================================================================
# 3. 기존 Rawdata 처리기 (Fallback 용도)
# ==================================================================================
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

def force_resample_data(df, target_fs=256, expected_duration_sec=196):
    """ 데이터가 부족할 때 강제로 196초 분량으로 늘리는 함수 """
    current_len = len(df)
    target_len = int(target_fs * expected_duration_sec) 
    
    if abs(current_len - target_len) / target_len < 0.1:
        return df # 정상이면 패스
    
    print(f"⚠️ 데이터 길이 보정 실행: {current_len}행 -> {target_len}행 (목표: {expected_duration_sec}초)")
    
    old_indices = np.linspace(0, 1, current_len)
    new_indices = np.linspace(0, 1, target_len)
    new_df = pd.DataFrame()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            new_df[col] = np.interp(new_indices, old_indices, df[col].values)
            
    return new_df

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

def analyze_concentration_intervals(file_path, baseline_sec=30.0, z_threshold=0.7, z_ceiling=4.0):
    print(f"📂 Rawdata 분석 모드 실행: {file_path}")
    try:
        try:
            df = pd.read_csv(file_path, delimiter="\t", encoding='cp949')
        except:
            df = pd.read_csv(file_path, delimiter="\t", encoding='utf-8')
        if df.empty: return []
        # ============================================================
        # [중요] 영상 길이 + 30초 = 196초로 고정
        VIDEO_DURATION_SEC = 196     # 영상  길이에 따라 +30초를 더해서 계산을 해야함. 현재 영상 길이는 2분 46초. 검은 화면 30초
        df = force_resample_data(df, target_fs=256, expected_duration_sec=VIDEO_DURATION_SEC)
        # ============================================================
        
        fs = 256
        processor = SignalProcessor(fs)
        
        col_fp1 = next((c for c in df.columns if 'Fp1' in c), df.columns[1])
        col_fp2 = next((c for c in df.columns if 'Fp2' in c), df.columns[2])
        signal_fp1 = df[col_fp1].values
        signal_fp2 = df[col_fp2].values
        
        idx_fp1 = calculate_concentration_index(processor, signal_fp1)
        idx_fp2 = calculate_concentration_index(processor, signal_fp2)
        
        base_samples = int(baseline_sec * fs)
        
        # 안전장치: 데이터가 baseline보다 짧을 경우
        if len(idx_fp1) <= base_samples:
            base_fp1 = idx_fp1
            base_fp2 = idx_fp2
        else:
            base_fp1 = idx_fp1[:base_samples]
            base_fp2 = idx_fp2[:base_samples]
        
        std_fp1 = np.std(base_fp1) if np.std(base_fp1) > 1e-10 else 1.0
        std_fp2 = np.std(base_fp2) if np.std(base_fp2) > 1e-10 else 1.0
        
        z_fp1 = (idx_fp1 - np.mean(base_fp1)) / std_fp1
        z_fp2 = (idx_fp2 - np.mean(base_fp2)) / std_fp2
        
        # 시간 축 생성 (0 ~ 196초)
        time_axis = np.linspace(0, len(z_fp1)/fs, len(z_fp1))
        
        save_z_score_plot(time_axis, z_fp1, z_fp2, z_threshold, baseline_sec, mode="Rawdata")
        if len(z_fp1) > base_samples:
            task_z_fp1 = z_fp1[base_samples:]
            task_z_fp2 = z_fp2[base_samples:]
        else:
            return []
        is_active = ((task_z_fp1 > z_threshold) | (task_z_fp2 > z_threshold))
        
        intervals = []
        start = None
        
        for i, active in enumerate(is_active):
            curr_task_time = i / fs
            if active and start is None:
                start = curr_task_time
            elif not active and start is not None:
                if curr_task_time - start >= 3.0: 
                    intervals.append((start + baseline_sec, curr_task_time + baseline_sec))
                start = None
                
        if start is not None:
            end_task_time = len(is_active)/fs
            if end_task_time - start >= 3.0:
                 intervals.append((start + baseline_sec, end_task_time + baseline_sec))
        if not intervals:
            save_analysis_log(["집중 구간 없음 (그래프 확인 요망)"])
            return []
        return intervals
    except Exception as e:
        print(f"🚨 Rawdata 분석 오류: {e}")
        return []

# ==================================================================================
# 4. [메인 진입점] 파일 탐색 및 실행 관리
# ==================================================================================
def get_latest_rawdata_path(base_path="C:/MAVE_RawData"):
    if os.path.exists(base_path):
        try:
            all_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if all_folders:
                latest_folder_path = max(all_folders, key=os.path.getctime)
                return os.path.join(latest_folder_path, "Rawdata.txt")
        except: pass
    return None

def load_timestamp_durations_from_file(file_path=None, ignored=None):
    # 1. 대상 파일 경로 확보
    if file_path and os.path.exists(file_path):
        target_path = file_path
    else:
        target_path = get_latest_rawdata_path()
    
    if not target_path:
        print("🚨 유효한 Rawdata.txt 파일을 찾을 수 없습니다.")
        return []
    # 2. 같은 폴더에 Biomarkers.txt가 있는지 확인
    folder_path = os.path.dirname(target_path)
    biomarker_path = os.path.join(folder_path, "Biomarkers.txt")
    
    # 3. 우선순위: Biomarkers.txt가 존재하면 최우선으로 사용!
    # (데이터 유실 상황에서도 값이 정확하므로 Rawdata보다 품질이 좋음)
    if os.path.exists(biomarker_path):
        result = analyze_biomarkers(biomarker_path, video_duration_sec=196, z_threshold=0.7)
        if result is not None:
            return result
        else:
            print("⚠️ Biomarkers 분석 실패, Rawdata 분석으로 넘어갑니다.")
    # 4. 없으면 기존 방식(Rawdata 리샘플링) 사용
    return analyze_concentration_intervals(target_path, z_threshold=0.7)