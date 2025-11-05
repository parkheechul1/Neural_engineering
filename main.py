import sys
from PyQt5.QtWidgets import QApplication
# 앞으로 만들 neural_engineering 폴더에서 UI 클래스를 가져옵니다.
from neural_engineering.main_window import SummaryApp

if __name__ == '__main__':
    """
    프로그램 실행 전용 파일   
    """
    app = QApplication(sys.argv)

    # AI 모델 로딩 실패 시 GUI가 알림창을 띄워줄 것입니다.
    ex = SummaryApp()
    ex.show()

    # 프로그램 종료 시 SummaryApp의 closeEvent가 실행되어 스레드를 정리합니다.
    sys.exit(app.exec_())