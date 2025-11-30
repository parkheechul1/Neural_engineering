import sys
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
            total_tasks = len(durations)

            if total_tasks == 0:
                raise Exception("집중 구간이 없습니다.")

            # ▼▼▼ [핵심] 로그 파일에 구분선 추가 ▼▼▼
            try:
                with open("analysis_log.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + "="*40 + "\n")
                    f.write(f"   [AI 내용 분석 결과] (총 {total_tasks}개 구간)\n")
                    f.write("="*40 + "\n")
            except Exception as e:
                print(f"로그 파일 열기 실패: {e}")
            # ▲▲▲ -------------------------------- ▲▲▲

            # 2. 구간별 AI 요약 실행
            for i, (start_sec, end_sec) in enumerate(durations):
                if not self._is_running: break

                timestamp_str = f"{start_sec:.2f} s - {end_sec:.2f} s"
                
                # Gemini + Whisper 실행
                full_text, summary_text = summarize_audio_duration(self.video_path, start_sec, end_sec)

                # ▼▼▼ [핵심] 요약된 내용을 로그 파일에 덧붙여 쓰기 (Append) ▼▼▼
                try:
                    with open("analysis_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n⏰ 구간: {timestamp_str}\n")
                        f.write(f"   🗣️ 원본: {full_text}\n")
                        f.write(f"   📝 요약: {summary_text}\n")
                        f.write("-" * 30 + "\n")
                except Exception as e:
                    print(f"로그 작성 실패: {e}")
                # ▲▲▲ -------------------------------------------------- ▲▲▲

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

        fileName, _ = QFileDialog.getOpenFileName(self, "파일 선택 (아무거나)", "", "Text Files (*.txt)")
        if fileName == '': return

        self.current_timestamps_path = fileName
        self.loadTimestampButton.setEnabled(False)
        self.loadTimestampButton.setText("분석 및 요약 생성 중...")
        self.timestampList.clear()
        self.summaries = {}
        self.summaryProgressBar.setValue(0)
        self.summaryProgressBar.setVisible(True)

        self.worker_thread = Worker(self.current_video_path, self.current_timestamps_path, self.z_threshold)
        self.worker_thread.summaryReady.connect(self.onSummaryReady)
        self.worker_thread.progressUpdated.connect(self.onProgressUpdated)
        self.worker_thread.errorOccurred.connect(self.onErrorOccurred)
        self.worker_thread.finished.connect(self.onWorkerFinished)
        self.worker_thread.start()

    def onSummaryReady(self, timestamp_str, summary_text, full_text):
        item_text = f"[{timestamp_str}] {summary_text}"
        self.timestampList.addItem(item_text)
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