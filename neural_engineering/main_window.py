import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QTextEdit,
                             QTabWidget, QStyle, QSlider, QLabel, QMessageBox, QProgressBar)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal

# 만든 모듈들 불러오기
from .eeg_handler import load_timestamp_durations_from_file
from .video_analyzer import summarize_audio_duration, get_ai_models


class Worker(QThread):
    summaryReady = pyqtSignal(str, str, str)
    progressUpdated = pyqtSignal(int, int)
    errorOccurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, video_path, timestamp_path, z_threshold):
        super().__init__()
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.z_threshold = z_threshold
        self._is_running = True

    def run(self):
        try:
            stt_model, summarizer_model = get_ai_models()
            if not stt_model:
                raise Exception("AI 모델 로드 실패")

            # 1. 뇌파 분석 실행 (여기서 1차 로그가 저장됨)
            durations = load_timestamp_durations_from_file(self.timestamp_path, self.z_threshold)
            
            # [필터링] 30초 이후에 시작된 구간만 남기기 (초반 멍 때리기 구간 제외)
            valid_durations = [d for d in durations if d[0] >= 30.0]
            total_tasks = len(valid_durations)

            if total_tasks == 0:
                # 30초 이후에 잡힌 게 없으면 안내 메시지
                print("30초 이후 유효한 집중 구간이 없습니다.")
                self.finished.emit()
                return

             # 로그 헤더 작성
            with open("analysis_log.txt", "a", encoding="utf-8") as f:
                f.write("\n" + "="*40 + "\n")
                f.write(f"   [AI 심층 분석] (유효 구간: {total_tasks}개)\n")
                f.write(f"   *전략: 초반 30초 제외 + 앞뒤 5초 문맥 확보\n")
                f.write("="*40 + "\n")   

            # 2. 구간별 AI 요약 실행
            for i, (start_sec, end_sec) in enumerate(valid_durations):
                if not self._is_running: break

                # ▼▼▼ [핵심] 앞뒤 5초씩 살 붙이기 (Padding) ▼▼▼
                # 시작은 0초보다 작아질 수 없으므로 max 사용
                padded_start = max(0, start_sec - 5.0)
                # 끝은 영상 길이를 넘을 수 없지만, video_analyzer에서 알아서 잘라줌
                padded_end = end_sec + 5.0
                # ▲▲▲ --------------------------------------- ▲▲▲

                timestamp_str = f"{start_sec:.2f} s - {end_sec:.2f} s"
                
                # Gemini에게는 '넉넉한 시간(padded)'을 줍니다.
                full_text, summary_text = summarize_audio_duration(self.video_path, padded_start, padded_end)

                # 로그 저장
                try:
                    with open("analysis_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n⏰ 핵심 구간: {timestamp_str} (분석: {padded_start:.1f}~{padded_end:.1f}s)\n")
                        f.write(f"   🗣️ 원본(확장): {full_text}\n")
                        f.write(f"   📝 요약: {summary_text}\n")
                        f.write("-" * 30 + "\n")
                except Exception as e:
                    print(f"로그 작성 실패: {e}")

                self.summaryReady.emit(timestamp_str, summary_text, full_text)
                self.progressUpdated.emit(i + 1, total_tasks)

        except Exception as e:
            self.errorOccurred.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False


class SummaryApp(QWidget):
    def __init__(self, z_threshold=1.0):
        super().__init__()
        self.z_threshold = z_threshold
        
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.timestampList = QListWidget()
        self.summaryEdit = QTextEdit()
        self.fullTextEdit = QTextEdit()
        self.summaries = {}
        self.current_video_path = None
        self.current_timestamps_path = None
        self.worker_thread = None

        self.initUI()
        self.setWindowTitle(f'뇌파 집중구간 오디오 요약 (Threshold: {self.z_threshold})')

    def loadVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "Video Files (*.mp4 *.avi *.mkv)")
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.current_video_path = fileName
            self.mediaPlayer.play()

    def initUI(self):
        mainLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()

        # 왼쪽: 비디오
        leftLayout.addWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        controlLayout = QHBoxLayout()
        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.playPause)
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        leftLayout.addLayout(controlLayout)

        # 오른쪽: 컨트롤 및 결과
        self.loadVideoButton = QPushButton("1. 영상 불러오기 (.mp4 등)")
        self.loadTimestampButton = QPushButton("2. 뇌파 데이터 불러오기 (자동 요약 시작)")
        self.summaryProgressBar = QProgressBar(self)
        self.summaryProgressBar.setVisible(False)

        rightLayout.addWidget(self.loadVideoButton)
        rightLayout.addWidget(self.loadTimestampButton)
        rightLayout.addWidget(self.summaryProgressBar)
        rightLayout.addWidget(QLabel("요약 결과 (클릭 시 확인):"))
        self.timestampList.setWordWrap(True)
        rightLayout.addWidget(self.timestampList)

        self.summaryTabs = QTabWidget()
        
        # 텍스트 색상 강제 지정 (화면 안 보이는 문제 방지)
        self.summaryEdit.setStyleSheet("QTextEdit { color: black; background-color: white; font-size: 14px; }")
        self.fullTextEdit.setStyleSheet("QTextEdit { color: black; background-color: white; font-size: 14px; }")

        self.summaryEdit.setReadOnly(True)
        self.fullTextEdit.setReadOnly(True)
        self.summaryTabs.addTab(self.summaryEdit, "AI 요약 (Gemini)")
        self.summaryTabs.addTab(self.fullTextEdit, "전체 텍스트 (Whisper)")
        rightLayout.addWidget(self.summaryTabs)

        mainLayout.addLayout(leftLayout, 2)
        mainLayout.addLayout(rightLayout, 1)
        self.setLayout(mainLayout)
        self.setGeometry(100, 100, 1200, 700)

        self.loadVideoButton.clicked.connect(self.loadVideo)
        self.loadTimestampButton.clicked.connect(self.loadTimestamps)
        self.timestampList.currentItemChanged.connect(self.jumpToTimestamp)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

    def loadTimestamps(self):
        if not self.current_video_path:
            QMessageBox.warning(self, "오류", "먼저 '1. 영상 불러오기'를 실행해주세요.")
            return
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "처리 중", "작업 진행 중입니다.")
            return

        # [수정됨] 폴더 선택 다이얼로그로 변경
        # 기본 경로는 C:/MAVE_RawData 로 설정 (없으면 현재 폴더)
        default_dir = "C:/MAVE_RawData" if os.path.exists("C:/MAVE_RawData") else ""
        folder_path = QFileDialog.getExistingDirectory(self, "뇌파 데이터 폴더 선택", default_dir)

        if folder_path == '': return  # 취소 누름

        # 선택한 폴더 내에 Rawdata.txt가 있는지 확인
        target_file_path = os.path.join(folder_path, "Rawdata.txt")

        if not os.path.exists(target_file_path):
            QMessageBox.critical(self, "파일 없음", f"선택한 폴더에 'Rawdata.txt' 파일이 없습니다.\n경로: {target_file_path}")
            return

        # 경로 확정
        self.current_timestamps_path = target_file_path
        
        self.loadTimestampButton.setEnabled(False)
        self.loadTimestampButton.setText("분석 및 요약 생성 중...")
        self.timestampList.clear()
        self.summaries = {}
        self.summaryProgressBar.setValue(0)
        self.summaryProgressBar.setVisible(True)

        # Worker에게 'Rawdata.txt'의 전체 경로를 넘김
        self.worker_thread = Worker(self.current_video_path, self.current_timestamps_path, self.z_threshold)
        self.worker_thread.summaryReady.connect(self.onSummaryReady)
        self.worker_thread.progressUpdated.connect(self.onProgressUpdated)
        self.worker_thread.errorOccurred.connect(self.onErrorOccurred)
        self.worker_thread.finished.connect(self.onWorkerFinished)
        self.worker_thread.start()

    def onSummaryReady(self, timestamp_str, summary_text, full_text):
        # [수정 후] 위쪽 리스트에는 '복원된 원본 문장(full_text)'을 표시
        item_text = f"[{timestamp_str}] {full_text}"
        
        self.timestampList.addItem(item_text)
        
        # 데이터 저장 (이 부분은 그대로 둠)
        # 키: 타임스탬프, 값: (요약, 원본) 튜플
        self.summaries[timestamp_str] = (summary_text, full_text)

    def onProgressUpdated(self, value, total):
        self.summaryProgressBar.setRange(0, total)
        self.summaryProgressBar.setValue(value)

    def onErrorOccurred(self, error_message):
        QMessageBox.critical(self, "오류", error_message)
        self.onWorkerFinished()

    def onWorkerFinished(self):
        self.loadTimestampButton.setEnabled(True)
        self.loadTimestampButton.setText("2. 뇌파 데이터 불러오기")
        self.summaryProgressBar.setVisible(False)

    def jumpToTimestamp(self, current_item, previous_item):
        if current_item is None: return
        item_text = current_item.text()
        
        timestamp_str = ""
        try:
            timestamp_str = item_text[item_text.find("[")+1 : item_text.find("]")]
            key = timestamp_str.strip()
            
            # [수정] 텍스트 먼저 표시 (안전장치)
            summary_tuple = self.summaries.get(key)
            if summary_tuple:
                self.summaryEdit.setText(summary_tuple[0])
                self.fullTextEdit.setText(summary_tuple[1])
            else:
                self.summaryEdit.setText("내용 없음")
                self.fullTextEdit.setText("")
                
        except Exception as e:
            print(f"GUI 오류: {e}")

        # 비디오 이동 (에러 무시)
        try:
            if timestamp_str:
                start_sec = float(timestamp_str.split(' ')[0])
                self.mediaPlayer.setPosition(int(start_sec * 1000))
                self.mediaPlayer.pause()
        except: pass

    def playPause(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState: self.mediaPlayer.pause()
        else: self.mediaPlayer.play()
    def mediaStateChanged(self, state):
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if state == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay))
    def positionChanged(self, position): self.positionSlider.setValue(position)
    def durationChanged(self, duration): self.positionSlider.setRange(0, duration)
    def setPosition(self, position): self.mediaPlayer.setPosition(position)
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        event.accept()