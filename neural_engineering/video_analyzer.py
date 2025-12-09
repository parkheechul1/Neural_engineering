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
        model="openai/whisper-small",
        device=device
    )
    print("✅ Whisper(STT) 모델 로드 완료.")
    print("✅ Gemini API 연결 준비 완료.")

except Exception as e:
    stt_pipeline = None
    print(f"🚨 AI 모델 로드 실패: {e}")

def get_ai_models():
    return stt_pipeline, "Gemini-API"

def summarize_with_gemini(full_text):
    if not GOOGLE_API_KEY: return "(API 키 오류)"

    # ✅ 수정 1: 무료 티어에서 가장 안정적인 'gemini-1.5-flash'를 메인으로 사용
    # 수정 코드 (1.5 -> 2.5 로 변경)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    
    final_prompt = f"{SYSTEM_PROMPT}\n\n[입력 텍스트]\n{full_text}"
    payload = {"contents": [{"parts": [{"text": final_prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            try:
                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            except KeyError:
                return "(응답 형식 오류 - 안전 필터 등에 걸렸을 수 있음)"
        
        # ✅ 수정 2: 에러 코드를 명확하게 반환하여 디버깅 돕기
        error_msg = f"(API 에러: {response.status_code})"
        try:
            # 구글이 보내준 상세 에러 메시지가 있다면 같이 표시
            error_json = response.json()
            if 'error' in error_json:
                error_msg += f" {error_json['error'].get('message', '')}"
        except:
            pass
            
        print(f"🚨 Gemini 호출 실패: {error_msg}") # 콘솔에도 출력
        return error_msg

    except Exception as e:
        return f"(통신 실패: {e})"

# (summarize_with_fallback 함수는 이제 필요 없으므로 삭제하거나 두셔도 됩니다)

def summarize_audio_duration(video_path, start_sec, end_sec):
    if not stt_pipeline: return "STT 모델 없음", "요약 불가"

    try:
        with VideoFileClip(video_path) as video:
            
            #[수정된 부분] 실제 영상 길이를 넘지 않도록 보정
            if end_sec > video.duration:
                end_sec = video.duration
            # (안전장치) 보정 후 시작 시간이 종료 시간보다 크거나 같으면 처리 중단
            if start_sec >= end_sec:
                return "(구간 오류)", "(영상 끝부분이라 요약할 내용이 없음)"
            audio_clip = video.subclip(start_sec, end_sec).audio
            # 오디오가 없는 경우 처리
            if audio_clip is None:
                 return "(오디오 없음)", "(내용 없음)"
            
            audio_array = audio_clip.to_soundarray(fps=16000)
            if audio_array.ndim > 1: audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)

        if len(audio_array) == 0: return "(무음)", "(내용 없음)"

        result = stt_pipeline(audio_array, chunk_length_s=30, return_timestamps=False)
        full_text = result['text'].strip()

        if not full_text: return "(음성 없음)", "(내용 없음)"

        if len(full_text) < 5: 
            summary_text = full_text # 너무 짧으면 요약 안 함
        else: 
            summary_text = summarize_with_gemini(full_text)

        return full_text, summary_text 

    except Exception as e:
        print(f"오디오 처리 중 오류: {e}")
        return f"오류: {e}", "오류"