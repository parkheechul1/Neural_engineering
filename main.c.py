import sys
import cv2  # OpenCV for frame extraction
from PIL import Image  # For image processing

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QTextEdit,
                             QVideoWidget, QStyle, QSlider, QLabel, QMessageBox)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt

# --- AI 모델 로딩 (애플리케이션 시작 시) ---
# Hugging Face Transformers 라이브러리 로드
try:
    from transformers import pipeline
    # "image-to-text" (이미지 캡셔닝) 파이프라인 로드
    # 모델: Salesforce/blip-image-captioning-large
    # 처음 실행 시 모델 파일을 다운로드하므로 시간이 걸릴 수 있습니다.
    image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    print("AI 이미지 캡셔닝 모델 로드 완료.")
except Exception as e:
    image_captioning_pipeline = None
    print(f"AI 모델 로드 실패: {e}. AI 요약 기능이 비활성화됩니다.")
    # (GUI 시작 전이므로 QMessageBox 대신 print 사용)
# --- AI 모델 로딩 완료 ---


class SummaryApp(QWidget):
    def __init__(self):
        super().__init__()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.timestampList = QListWidget()
        self.summaryEdit = QTextEdit()

        self.summaries = {}
        self.current_video_path = None  # 현재 로드된 영상 파일 경로 저장

        self.initUI()

        # AI 모델 로드 실패 시 사용자에게 알림
        if not image_captioning_pipeline:
            QMessageBox.warning(self, "AI 모델 로드 실패",
                                "AI 이미지 캡셔닝 모델을 로드하지 못했습니다. AI 요약 기능이 동작하지 않을 수 있습니다. "
                                "인터넷 연결을 확인하거나 필요한 라이브러리(torch)를 설치하세요.")


    def initUI(self):
        mainLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()

        # --- 왼쪽: 비디오 플레이어 ---
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

        # --- 오른쪽: 컨트롤 패널 ---
        self.loadVideoButton = QPushButton("1. 영상 불러오기 (.mp4 등)")
        self.loadTimestampButton = QPushButton("2. 타임스탬프 불러오기 (.txt)")
        self.autoSummarizeButton = QPushButton("3. AI로 자동 요약하기")
        self.saveSummaryButton = QPushButton("현재 내용 수동 저장")

        rightLayout.addWidget(self.loadVideoButton)
        rightLayout.addWidget(self.loadTimestampButton)
        rightLayout.addWidget(QLabel("집중 구간 (클릭 시 해당 시간으로 이동):"))
        rightLayout.addWidget(self.timestampList)
        rightLayout.addWidget(QLabel("내용 요약 (AI가 자동 생성):"))
        rightLayout.addWidget(self.summaryEdit)
        rightLayout.addWidget(self.autoSummarizeButton) # AI 버튼
        rightLayout.addWidget(self.saveSummaryButton)

        # AI 모델 로드 실패 시 버튼 비활성화
        if not image_captioning_pipeline:
            self.autoSummarizeButton.setEnabled(False)
            self.autoSummarizeButton.setText("3. AI 요약 (비활성화됨)")

        mainLayout.addLayout(leftLayout, 2)
        mainLayout.addLayout(rightLayout, 1)

        self.setLayout(mainLayout)
        self.setWindowTitle('영상 구간 AI 자동 요약 프로그램 (Demo)')
        self.setGeometry(100, 100, 1200, 700)

        # --- 시그널 연결 ---
        self.loadVideoButton.clicked.connect(self.loadVideo)
        self.loadTimestampButton.clicked.connect(self.loadTimestamps)
        self.saveSummaryButton.clicked.connect(self.saveSummary)
        self.autoSummarizeButton.clicked.connect(self.generateAISummary) # AI 버튼 시그널
        self.timestampList.currentItemChanged.connect(self.jumpToTimestamp)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

    # --- 핵심 기능 함수 ---

    def loadVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "Video Files (*.mp4 *.avi *.mkv)")
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.current_video_path = fileName # 영상 경로 저장!
            print(f"영상 로드됨: {self.current_video_path}")

    def loadTimestamps(self):
        # 5번 데모를 위해, 뇌파 분석(4번) 대신 수동 타임스탬프 파일을 사용
        fileName, _ = QFileDialog.getOpenFileName(self, "타임스탬프 파일 선택", "", "Text Files (*.txt)")
        if fileName != '':
            self.timestampList.clear()
            self.summaries = {}
            try:
                with open(fileName, 'r') as f:
                    for line in f:
                        try:
                            timestamp_sec = float(line.strip())
                            item_text = f"{timestamp_sec:.2f} s"
                            self.timestampList.addItem(item_text)
                            self.summaries[item_text] = "" # 요약 비워두기
                        except ValueError:
                            pass # 빈 줄이나 잘못된 형식 무시
                print(f"타임스탬프 로드됨: {fileName}")
            except Exception as e:
                QMessageBox.critical(self, "파일 오류", f"타임스탬프 파일 읽기 실패: {e}")

    def jumpToTimestamp(self, current_item, previous_item):
        if current_item is None: return
        timestamp_str = current_item.text()
        try:
            timestamp_sec = float(timestamp_str.split(' ')[0])
            position_ms = int(timestamp_sec * 1000)
            self.mediaPlayer.setPosition(position_ms)
            self.mediaPlayer.pause()
            self.summaryEdit.setText(self.summaries.get(timestamp_str, "")) # 저장된 요약 불러오기
        except Exception as e:
            print(f"시간 이동 오류: {e}")

    def saveSummary(self):
        currentItem = self.timestampList.currentItem()
        if currentItem:
            timestamp_str = currentItem.text()
            summary_text = self.summaryEdit.toPlainText()
            self.summaries[timestamp_str] = summary_text
            QMessageBox.information(self, "저장", f"[{timestamp_str}]의 요약이 저장되었습니다.")

    def generateAISummary(self):
        """[핵심 기능] AI를 사용하여 현재 선택된 타임스탬프의 영상 내용을 자동 요약"""
        if not image_captioning_pipeline: return

        currentItem = self.timestampList.currentItem()
        if not currentItem:
            QMessageBox.warning(self, "선택 오류", "먼저 타임스탬프 목록에서 항목을 선택해주세요.")
            return

        if not self.current_video_path:
            QMessageBox.warning(self, "영상 없음", "먼저 영상을 로드해주세요.")
            return

        timestamp_str = currentItem.text()
        try:
            timestamp_sec = float(timestamp_str.split(' ')[0])

            self.summaryEdit.setText("AI가 장면을 분석 중입니다... 잠시만 기다려주세요...")
            QApplication.processEvents() # UI가 멈추지 않도록 강제 업데이트

            # 1. OpenCV로 비디오 파일 열기
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                raise Exception(f"OpenCV가 영상을 열 수 없습니다: {self.current_video_path}")

            # 2. 해당 시간대의 프레임으로 이동
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read() # 프레임 읽기
            cap.release() # 비디오 캡처 객체 해제

            if ret:
                # 3. OpenCV(BGR) -> PIL(RGB) 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # 4. AI 모델 호출 (Hugging Face Pipeline)
                # 이 부분은 GPU가 없으면 몇 초간 멈출 수 있습니다.
                ai_output = image_captioning_pipeline(image)

                # 5. 결과 파싱 및 UI 업데이트
                # 결과 예: [{'generated_text': 'a dog playing fetch'}]
                generated_text = ai_output[0]['generated_text'] if ai_output else "AI 요약 실패"

                self.summaryEdit.setText(generated_text)
                self.summaries[timestamp_str] = generated_text # AI 요약 결과도 저장
                print(f"AI 자동 요약: [{timestamp_str}] -> {generated_text}")
            else:
                self.summaryEdit.setText("오류: 해당 시간대의 프레임을 추출할 수 없습니다.")

        except Exception as e:
            self.summaryEdit.setText(f"AI 요약 중 오류 발생: {e}")
            print(f"AI 요약 중 오류 발생: {e}")

    # --- 미디어 플레이어 컨트롤 함수들 (이전과 동일) ---
    def playPause(self):
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SummaryApp()
    ex.show()
    sys.exit(app.exec_())