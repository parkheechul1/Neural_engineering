import os
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import requests
import json
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT

# 1. .env 파일 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("🚨 [오류] .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")
else:
    print(f"🔑 API 키 로드 성공 (앞 5자리: {GOOGLE_API_KEY[:5]}...)")

# --- AI 모델 로딩 (Whisper) ---
try:
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"음성 인식(STT) 장치: {device}")

    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base", 
        device=device
    )
    print("✅ Whisper(STT) 모델 로드 완료.")
    print("✅ Gemini API 연결 준비 완료.")

except Exception as e:
    stt_pipeline = None
    print(f"🚨 AI 모델 로드 실패: {e}")

# ▼▼▼ [수정된 부분] 여기가 에러 원인이었습니다! ▼▼▼
def get_ai_models():
    # 예전 코드: return stt_pipeline, summarizer_pipeline (X - 에러 발생)
    # 수정 코드: return stt_pipeline, "Gemini-API" (O - 정상)
    return stt_pipeline, "Gemini-API"
# ▲▲▲ --------------------------------------- ▲▲▲

def find_available_gemini_model():
    """사용 가능한 모델 자동 탐색"""
    if not GOOGLE_API_KEY: return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models:
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']:
                    if 'flash' in m['name'] or '1.5' in m['name']: return m['name']
            for m in models:
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']:
                    return m['name']
    except: pass
    return "models/gemini-pro"

def summarize_with_gemini(full_text):
    if not GOOGLE_API_KEY: return "(API 키 오류)"

    model_name = find_available_gemini_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GOOGLE_API_KEY}"
    
    final_prompt = f"{SYSTEM_PROMPT}\n\n[입력 텍스트]\n{full_text}"
    payload = {"contents": [{"parts": [{"text": final_prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"(API 에러: {response.status_code})"
    except Exception as e:
        return f"(통신 실패: {e})"

def summarize_audio_duration(video_path, start_sec, end_sec):
    if not stt_pipeline: return "STT 모델 없음", "요약 불가"

    try:
        with VideoFileClip(video_path) as video:
            audio_clip = video.subclip(start_sec, end_sec).audio
            audio_array = audio_clip.to_soundarray(fps=16000)
            if audio_array.ndim > 1: audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)

        if len(audio_array) == 0: return "(무음)", "(내용 없음)"

        result = stt_pipeline(audio_array, chunk_length_s=30, return_timestamps=False)
        full_text = result['text'].strip()

        if not full_text: return "(음성 없음)", "(내용 없음)"

        if len(full_text) < 5: summary_text = full_text
        else: summary_text = summarize_with_gemini(full_text)

        return full_text, summary_text 

    except Exception as e:
        return f"오류: {e}", "오류"