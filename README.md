신경공학 팀플


```bash
#아래 pip 모두 실행 후 진행
pip install -r requirements.txt

#혹시 기존에 깔린 게 있다면 삭제
pip uninstall torch -y

#CPU 버전으로 재설치 (그래픽카드 오류 방지)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
```
1. 3-4분정도 되는 강의 영상 시청.

2. 사람이 뇌파장치를 착용하고 위 영상을 관찰한다.

3.1. 뇌파 신호를 처리할 수 있는 데이터 형태로 변형
3.2 영상이 끝나면 착용자의 뇌파를 분석한 데이터를 가짐.

4.1 영상을 보는 과정 중에서 "집중"하고 있다고 파악되는 부분이 있을 것임.

5. 4번에서 파악한 부분을 자동으로 요약해줌
```


# Neural Engineering Project: 뇌파 기반 중요 학습 구간 자동 요약 시스템

## 1. 프로젝트 개요
본 프로젝트는 사용자가 영상을 시청하는 동안 발생하는 뇌파(EEG)를 분석하여, 인지적으로 집중한 순간(Highlight)을 자동으로 탐지합니다. 탐지된 구간의 오디오를 텍스트로 변환(STT)하고 AI 요약을 수행하여 학습 효율을 극대화하는 시스템입니다.

---

## 2. 시스템 동작 원리 (Algorithm Logic)

본 시스템은 **오프라인 분석(Offline Analysis)** 방식을 채택하여, 녹화된 뇌파 데이터를 정밀 분석합니다.

### Step 1. 데이터 수집 및 자동 매칭
- 프로그램은 `C:/MAVE_RawData` 경로를 탐색하여 **가장 최근에 생성된** 뇌파 데이터 폴더를 자동으로 찾아냅니다.
- 별도의 파일 선택 없이, 사용자가 방금 측정한 데이터를 즉시 분석합니다.

### Step 2. 신호 처리 (Signal Processing)
- **전처리:** 전두엽 채널(`Fp1`, `Fp2`) 데이터에 Bandpass Filter를 적용하여 노이즈를 제거합니다.
- **지표 산출:** 각 채널별로 다음 두 가지 집중도 지표를 계산하여 통합합니다.
  - **BA (Beta / Alpha):** 능동적 집중 및 각성 상태 반영
  - **BT (Beta / Theta):** 인지 부하 및 몰입도 반영
  - **최종 지수:** `Concentration Index = (BA + BT) / 2`

### Step 3. 개인화 보정 (Normalization) - **핵심 로직**
사람마다, 날짜마다 뇌파의 기본 세기가 다르기 때문에 **상대적 변화량**을 측정합니다.
1. **Baseline 설정:** 데이터의 **초반 30초**를 '휴식 구간(Resting State)'으로 정의합니다.
2. **Z-Score 변환:** 영상 시청 구간의 데이터를 휴식 구간의 통계(평균, 표준편차)를 이용해 정규화합니다.
   > *"평소 쉴 때보다 표준편차의 몇 배만큼 더 뇌파가 활성화되었는가?"*

### Step 4. 중요 구간 판정 (Thresholding)
- **판정 기준:** `Fp1` 또는 `Fp2` 채널 중 하나라도 설정된 **임계값(Z-Threshold)**을 초과하면 집중 상태로 간주합니다.
- **최소 지속 시간:** 순간적인 노이즈를 배제하기 위해, **3초 이상** 지속된 구간만 최종 저장합니다.

---

## 3. 사용 가이드 (How to Run)

### 필수 환경
- Python 3.8+
- Anaconda 환경 권장 (`conda activate neural_engineering`)
- 라이브러리: `PyQt5`, `pandas`, `numpy`, `scipy`, `transformers`, `torch`

### 실행 순서
1. **프로그램 실행:**
   ```bash
   python main.py


```
Neural_engineering/
├── main.py                 # 프로그램 진입점 (실행 파일)
├── README.md               # 프로젝트 설명서
├── analysis_log.txt        # 분석 결과 로직(자동생성)
├── z_score_graph.png       # 개인별 z_score 수정을 위한 보조 자료
├── Neural_engineering.iml
│ 
│   
└── neural_engineering/     # 패키지 폴더
    ├── main_window.py      # GUI 구성 및 이벤트 처리
    ├── video_analyzer.py   # AI 요약 (Whisper, KoBART)
    ├── prompts.py          # 문장 요약을 위한 요약 요청 제미나이 프롬프트 
    └── eeg_handler.py      # 뇌파 신호 처리 및 알고리즘 핵심

```