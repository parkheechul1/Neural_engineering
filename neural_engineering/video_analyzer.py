import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # DLL 오류 방지용

import torch
from moviepy.editor import VideoFileClip
import numpy as np

# --- AI 모델 로딩 (2개의 모델 로드) ---
try:
    from transformers import pipeline

    # 1. 음성 인식 (STT) 모델 (Whisper)
    # GPU 사용 가능 여부 확인
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"음성 인식(STT)에 사용할 장치: {device}")

    # Whisper 모델 로드 (크기가 큼, 처음 실행 시 다운로드 오래 걸림)
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base", # (테스트용 'base' 모델, 더 정확한건 'large')
        device=device
    )
    print("AI 음성 인식(STT) 모델 로드 완료.")

    # 2. 텍스트 요약 모델
    summarizer_pipeline = pipeline(
        "summarization",
        model="gogamza/kobart-base-v2" # (한국어 요약)
    )
    print("AI 텍스트 요약 모델 로드 완료.")

except Exception as e:
    stt_pipeline = None
    summarizer_pipeline = None
    print(f"AI 모델 로드 실패: {e}")
# --- AI 모델 로딩 완료 ---


def get_ai_models():
    """로드된 AI 모델 파이프라인을 반환합니다."""
    return stt_pipeline, summarizer_pipeline

def summarize_audio_duration(video_path, start_sec, end_sec):
    """
    영상 파일의 특정 오디오 구간을 추출, 텍스트로 변환, 요약합니다.
    """
    if not stt_pipeline or not summarizer_pipeline:
        raise Exception("AI 모델이 로드되지 않았습니다.")

    try:
        # 1. (영상 -> 오디오) moviepy로 비디오 클립 로드
        with VideoFileClip(video_path) as video:
            # 지정된 시간만큼 오디오 서브클립 추출
            audio_clip = video.subclip(start_sec, end_sec).audio

            # 오디오 데이터를 STT 파이프라인이 요구하는 numpy 배열로 변환
            # Whisper는 16kHz 샘플링 레이트가 필요함
            audio_array = audio_clip.to_soundarray(fps=16000)

            # 스테레오(채널 2개)라면 모노(채널 1개)로 변환
            if audio_array.ndim > 1 and audio_array.shape[1] == 2:
                audio_array = audio_array.mean(axis=1)

            # STT 파이프라인은 부동 소수점 배열을 선호
            audio_array = audio_array.astype(np.float32)

        if len(audio_array) == 0:
            return "(오디오 없음)", "(내용 없음)"

        # 2. (오디오 -> 텍스트) 음성 인식(STT) 실행
        # chunk_length_s=30 : 긴 오디오도 30초 단위로 잘라서 처리
        transcription_result = stt_pipeline(audio_array, chunk_length_s=30, return_timestamps=False)
        full_text = transcription_result['text'].strip()

        if not full_text:
            return "(음성 없음)", "(내용 없음)"

        # 3. (텍스트 -> 요약) 텍스트 요약 실행
        # 너무 짧은 텍스트는 요약이 의미 없거나 오류를 낼 수 있음
        if len(full_text.split()) < 20: # 20단어 미만은 요약하지 않고 원본 반환
            summary_text = full_text
        else:
            # BART 모델은 min_length/max_length가 필요
            summary_result = summarizer_pipeline(full_text, max_length=150, min_length=10, do_sample=False)
            summary_text = summary_result[0]['summary_text']

        return full_text, summary_text # (전체 텍스트, 요약 텍스트) 튜플 반환

    except Exception as e:
        print(f"오디오 분석 오류 (구간: {start_sec}-{end_sec}s): {e}")
        return f"오류: {e}", "오류 발생"