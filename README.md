신경공학 팀플

아래 세개 다운받고 시작
pip install -r requirements.txt
# 혹시 기존에 깔린 게 있다면 삭제
pip uninstall torch -y
# CPU 버전으로 재설치 (그래픽카드 오류 방지)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

1. 3-4분정도 되는 강의 영상 시청.

2. 사람이 뇌파장치를 착용하고 위 영상을 관찰한다.

3.1. 뇌파 신호를 처리할 수 있는 데이터 형태로 변형
3.2 영상이 끝나면 착용자의 뇌파를 분석한 데이터를 가짐.

4.1 영상을 보는 과정 중에서 "집중"하고 있다고 파악되는 부분이 있을 것임.

5. 4번에서 파악한 부분을 자동으로 요약해줌

