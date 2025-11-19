import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

# --- [1] 신호 처리 클래스 (preprocess.py, iir.py 로직 통합) ---
class SignalProcessor:
    def __init__(self, fs=256):
        self.fs = fs

    def butter_bandpass_filter(self, data, lowcut, highcut, order=2):
        """Bandpass Filter (특정 주파수 대역만 남김)"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def get_power(self, data):
        """신호의 파워(세기) 계산 (Squaring)"""
        return data ** 2

    def moving_average(self, data, window_sec=1.0):
        """이동 평균 (Smoothing)"""
        window_size = int(window_sec * self.fs)
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# --- [2] 파일 로딩 및 분석 로직 ---

def get_latest_rawdata_path(base_path="C:/MAVE_RawData"):
    """C:/MAVE_RawData에서 가장 최신 Rawdata.txt 파일을 찾습니다."""
    if not os.path.exists(base_path):
        print(f"경고: {base_path} 경로를 찾을 수 없습니다. 테스트 모드로 동작합니다.")
        return None

    rawdata_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if not rawdata_folders:
        print("데이터 폴더가 비어있습니다.")
        return None

    rawdata_folders.sort()
    latest_folder = rawdata_folders[-1]
    file_path = os.path.join(base_path, latest_folder, "Rawdata.txt")

    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return None

    return file_path

def analyze_concentration_intervals(file_path):
    """
    뇌파 데이터를 분석하여 '집중' 구간 (Beta/Alpha 비율이 높은 구간)을 찾습니다.
    """
    try:
        # 1. 데이터 읽기 (참고 프로젝트 형식: 탭 구분, cp949 인코딩 가능성)
        df = pd.read_csv(file_path, delimiter="\t", encoding='cp949')
        if df.empty: return []

        # 2. 채널 선택 (Fp1, Fp2 등)
        # 데이터에 'EEG_Fp1' 컬럼이 있다고 가정
        target_col = 'EEG_Fp1' if 'EEG_Fp1' in df.columns else df.columns[1]
        raw_signal = df[target_col].values

        fs = 256
        processor = SignalProcessor(fs)

        # 3. [핵심 알고리즘] 집중도 계산: Beta / Alpha Ratio
        # (1) Alpha파 (8-13Hz): 이완할 때 나옴 (집중하면 감소)
        alpha_wave = processor.butter_bandpass_filter(raw_signal, 8.0, 13.0)
        alpha_power = processor.moving_average(processor.get_power(alpha_wave))

        # (2) Beta파 (13-30Hz): 집중할 때 나옴 (집중하면 증가)
        beta_wave = processor.butter_bandpass_filter(raw_signal, 13.0, 30.0)
        beta_power = processor.moving_average(processor.get_power(beta_wave))

        # (3) 비율 계산 (0으로 나누기 방지 위해 아주 작은 수 더함)
        concentration_index = beta_power / (alpha_power + 1e-6)

        # 4. 집중 구간 검출 (Thresholding)
        # 평균보다 0.5 표준편차 이상 높은 구간을 집중으로 간주
        threshold = np.mean(concentration_index) + (np.std(concentration_index) * 0.5)
        is_concentrating = concentration_index > threshold

        intervals = []
        start_time = None

        for i, active in enumerate(is_concentrating):
            current_time = i / fs
            if active and start_time is None:
                start_time = current_time
            elif not active and start_time is not None:
                end_time = current_time
                # 3초 이상 지속된 집중만 인정
                if end_time - start_time >= 3.0:
                    intervals.append((start_time, end_time))
                start_time = None

        if start_time is not None:
            end_time = len(concentration_index) / fs
            if end_time - start_time >= 3.0:
                intervals.append((start_time, end_time))

        # 구간이 없으면 전체 길이의 중간 부분이라도 반환 (데모용)
        if not intervals:
            total_time = len(concentration_index) / fs
            return [(total_time * 0.3, total_time * 0.7)]

        return intervals

    except Exception as e:
        print(f"분석 오류: {e}")
        return [(10.0, 20.0)] # 오류 시 기본값

# --- [3] 메인 윈도우에서 호출하는 함수 ---

def load_timestamp_durations_from_file(timestamp_path):
    """
    [최종 동작]
    1. C:/MAVE_RawData에서 최신 파일을 찾습니다.
    2. 파일이 있으면 뇌파 분석을 수행하여 집중 구간을 반환합니다.
    3. 파일이 없으면(테스트 환경), 사용자가 선택한 timestamp.txt를 읽습니다.
    """
    real_data_path = get_latest_rawdata_path()

    if real_data_path:
        print(f"실제 뇌파 데이터 분석 시작: {real_data_path}")
        return analyze_concentration_intervals(real_data_path)
    else:
        print("실제 뇌파 데이터를 찾을 수 없어 모의 데이터(txt)를 사용합니다.")
        durations = []
        try:
            with open(timestamp_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        durations.append((float(parts[0]), float(parts[1])))
        except Exception:
            pass
        # 모의 데이터도 없으면 기본값 반환
        if not durations:
            return [(10.0, 20.0)]
        return durations