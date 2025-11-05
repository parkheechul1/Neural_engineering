def load_timestamp_durations_from_file(timestamp_path):
    """
    .txt 파일에서 (시작시간, 종료시간) 구간 목록을 읽어 리스트로 반환합니다.

    파일 형식 예시:
    10.5, 20.0
    30.5, 45.2
    """
    durations = []
    try:
        with open(timestamp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        parts = line.split(',')
                        start_sec = float(parts[0].strip())
                        end_sec = float(parts[1].strip())
                        if start_sec < end_sec:
                            durations.append((start_sec, end_sec))
                        else:
                            print(f"경고: 시작 시간이 종료 시간보다 늦습니다. 무시됨: {line}")
                    except (ValueError, IndexError):
                        print(f"경고: 잘못된 형식의 라인입니다. 무시됨: {line}")
        return durations
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {timestamp_path}")
        raise
    except Exception as e:
        print(f"타임스탬프 파일 읽기 오류: {e}")
        raise