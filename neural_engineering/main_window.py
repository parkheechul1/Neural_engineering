import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QTextEdit,
                             QTabWidget, QStyle, QSlider, QLabel, QMessageBox, QProgressBar)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal

# 1. 다른 모듈에서 필요한 함수들을 import (수정됨)
from .eeg_handler import load_timestamp_durations_from_file # (이름 변경)
from .video_analyzer import summarize_audio_duration, get_ai_models # (이름 변경)


# 2. Worker 스레드 클래스 (수정됨)
class Worker(QThread):
    # (타임스탬프, 요약결과, 전체텍스트) 3개를 전달하도록 수정
    summaryReady = pyqtSignal(str, str, str)
    progressUpdated = pyqtSignal(int, int) # (현재, 전체)
    errorOccurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, video_path, timestamp_path):
        super().__init__()
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self._is_running = True

    def run(self):
        """스레드 시작 시 이 함수가 '백그라운드'에서 실행됩니다."""
        try:
            stt_model, summarizer_model = get_ai_models()
            if not stt_model or not summarizer_model:
                raise Exception("AI 모델 중 하나 이상이 로드되지 않았습니다.")

            # 1. [EEG 핸들러]에게 (시작, 종료) 구간 목록 요청 (수정됨)
            durations = load_timestamp_durations_from_file(self.timestamp_path)
            total_tasks = len(durations)

            if total_tasks == 0:
                raise Exception("타임스탬프 파일에 유효한 (시작, 종료) 구간 데이터가 없습니다.")

            # 2. 각 구간별로 [AI 분석기]에게 요약 요청
            for i, (start_sec, end_sec) in enumerate(durations):
                if not self._is_running: # 중지 신호 확인
                    break

                timestamp_str = f"{start_sec:.2f} s - {end_sec:.2f} s"

                # [AI 분석기] 호출 (수정됨)
                full_text, summary_text = summarize_audio_duration(self.video_path, start_sec, end_sec)

                # 3. UI로 결과 전송 (시그널 인자 3개로 수정)
                self.summaryReady.emit(timestamp_str, summary_text, full_text)
                self.progressUpdated.emit(i + 1, total_tasks)

        except Exception as e:
            self.errorOccurred.emit(str(e)) # UI로 오류 전송
        finally:
            self.finished.emit() # UI로 작업 완료 신호 전송

    def stop(self):
        self._is_running = False


# 3. 메인 GUI 클래스 (수정됨)
class SummaryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.timestampList = QListWidget()
        self.summaryEdit = QTextEdit()

        # (타임스탬프 문자열: (요약, 전체텍스트)) 튜플을 저장
        self.summaries = {}
        self.current_video_path = None
        self.current_timestamps_path = None
        self.worker_thread = None

        self.initUI()

    def loadVideo(self):
        """1. 영상 불러오기 버튼 클릭 시 실행"""
        fileName, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "Video Files (*.mp4 *.avi *.mkv)")
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.current_video_path = fileName

    def initUI(self):
        mainLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()

        # --- 왼쪽: 비디오 플레이어 (변경 없음) ---
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

        # --- 오른쪽: 컨트롤 패널 (변경 없음) ---
        self.loadVideoButton = QPushButton("1. 영상 불러오기 (.mp4 등)")
        self.loadTimestampButton = QPushButton("2. 뇌파 데이터 불러오기 (자동 요약 시작)")
        self.summaryProgressBar = QProgressBar(self)
        self.summaryProgressBar.setVisible(False)

        rightLayout.addWidget(self.loadVideoButton)
        rightLayout.addWidget(self.loadTimestampButton)
        rightLayout.addWidget(self.summaryProgressBar)
        rightLayout.addWidget(QLabel("요약 결과 (클릭 시 해당 구간 시작점으로 이동):"))
        self.timestampList.setWordWrap(True)
        rightLayout.addWidget(self.timestampList)

        # 탭 위젯 추가 (요약본 / 전체 텍스트 분리)
        self.summaryTabs = QTabWidget()
        self.summaryEdit = QTextEdit() # 요약 탭
        self.fullTextEdit = QTextEdit() # 전체 텍스트 탭
        self.summaryEdit.setReadOnly(True)
        self.fullTextEdit.setReadOnly(True)

        self.summaryTabs.addTab(self.summaryEdit, "AI 요약본")
        self.summaryTabs.addTab(self.fullTextEdit, "전체 변환 텍스트")

        rightLayout.addWidget(self.summaryTabs)

        stt_model, summarizer_model = get_ai_models()
        if not stt_model or not summarizer_model:
            self.loadTimestampButton.setEnabled(False)
            self.loadTimestampButton.setText("2. AI 로드 실패 (기능 비활성화)")

        mainLayout.addLayout(leftLayout, 2)
        mainLayout.addLayout(rightLayout, 1)
        self.setLayout(mainLayout)
        self.setWindowTitle('뇌파 집중구간 오디오 요약 (Demo)')
        self.setGeometry(100, 100, 1200, 700)

        # --- 시그널 연결 (변경 없음) ---
        self.loadVideoButton.clicked.connect(self.loadVideo)
        self.loadTimestampButton.clicked.connect(self.loadTimestamps)
        self.timestampList.currentItemChanged.connect(self.jumpToTimestamp)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

    # --- loadTimestamps (스레드 시작) (수정됨) ---
    def loadTimestamps(self):
        """'뇌파 데이터 불러오기' 클릭 시 자동 요약 스레드 시작"""
        if not self.current_video_path:
            QMessageBox.warning(self, "오류", "먼저 '1. 영상 불러오기'를 실행해주세요.")
            return
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "처리 중", "이미 요약 작업이 진행 중입니다.")
            return

        fileName, _ = QFileDialog.getOpenFileName(self, "타임스탬프 파일 선택", "", "Text Files (*.txt)")
        if fileName == '':
            return

        self.current_timestamps_path = fileName

        self.loadTimestampButton.setEnabled(False)
        self.loadTimestampButton.setText("뇌파 분석 및 AI 요약 중...")
        self.timestampList.clear()
        self.summaries = {}
        self.summaryProgressBar.setValue(0)
        self.summaryProgressBar.setVisible(True)

        # 스레드 생성 및 시작
        self.worker_thread = Worker(self.current_video_path, self.current_timestamps_path)

        # [수정] 새 시그널(인자 3개)에 연결
        self.worker_thread.summaryReady.connect(self.onSummaryReady)
        self.worker_thread.progressUpdated.connect(self.onProgressUpdated)
        self.worker_thread.errorOccurred.connect(self.onErrorOccurred)
        self.worker_thread.finished.connect(self.onWorkerFinished)

        self.worker_thread.start()

    # --- QThread 시그널 처리 함수 (Slot) (수정됨) ---
    def onSummaryReady(self, timestamp_str, summary_text, full_text):
        """스레드로부터 요약 결과(인자 3개)가 도착하면 호출됨"""
        item_text = f"[{timestamp_str}] {summary_text}"
        self.timestampList.addItem(item_text)
        # (요약, 전체텍스트) 튜플을 저장
        self.summaries[timestamp_str] = (summary_text, full_text)

    def onProgressUpdated(self, value, total):
        self.summaryProgressBar.setRange(0, total)
        self.summaryProgressBar.setValue(value)

    def onErrorOccurred(self, error_message):
        QMessageBox.critical(self, "AI 요약 오류", error_message)
        self.onWorkerFinished(error=True)

    def onWorkerFinished(self, error=False):
        self.loadTimestampButton.setEnabled(True)
        self.loadTimestampButton.setText("2. 뇌파 데이터 불러오기 (자동 요약 시작)")
        self.summaryProgressBar.setVisible(False)
        if not error and self.summaryProgressBar.value() > 0:
            QMessageBox.information(self, "완료", "모든 집중 구간의 자동 요약이 완료되었습니다.")

    # --- jumpToTimestamp (수정됨) ---
    def jumpToTimestamp(self, current_item, previous_item):
        """목록 클릭 시 해당 구간 시작점으로 이동, 탭에 내용 표시"""
        if current_item is None: return
        item_text = current_item.text()
        try:
            # 예: "[10.50 s - 20.00 s] This is a summary"
            timestamp_str = item_text[item_text.find("[")+1 : item_text.find("]")]
            # "10.50 s - 20.00 s"

            start_sec_str = timestamp_str.split(' ')[0]
            timestamp_sec = float(start_sec_str)
            position_ms = int(timestamp_sec * 1000)

            self.mediaPlayer.setPosition(position_ms) # 영상 이동
            self.mediaPlayer.pause()

            # 딕셔너리에서 (요약, 전체텍스트) 튜플을 찾아 탭에 표시
            key_str = timestamp_str.strip() # "10.50 s - 20.00 s"
            summary_tuple = self.summaries.get(key_str)

            if summary_tuple:
                self.summaryEdit.setText(summary_tuple[0]) # 요약본
                self.fullTextEdit.setText(summary_tuple[1]) # 전체 텍스트
            else:
                self.summaryEdit.setText("")
                self.fullTextEdit.setText("")

        except Exception as e:
            print(f"시간 이동/내용 표시 오류: {e}")
            self.summaryEdit.setText("")
            self.fullTextEdit.setText("")

    # --- 미디어 플레이어 함수들 (변경 없음) ---
    def playPause(self):
        # (이하 코드는 이전과 동일)
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        event.accept()