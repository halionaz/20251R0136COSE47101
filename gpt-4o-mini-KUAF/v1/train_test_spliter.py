import json
import random
import os


def split_jsonl(input_jsonl_path, train_ratio=0.8, random_seed=42):
    """
    JSONL 파일을 랜덤하게 train/test로 분할하는 함수

    Args:
        input_jsonl_path (str): 입력 JSONL 파일 경로
        train_ratio (float): 훈련 데이터 비율 (기본값: 0.8)
        random_seed (int): 랜덤 시드 (재현 가능한 결과를 위해)
    """
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(random_seed)

    # JSONL 파일 읽기
    print(f"JSONL 파일 읽는 중: {input_jsonl_path}")

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_records = len(lines)
    print(f"총 레코드 수: {total_records}")

    # 랜덤하게 섞기
    random.shuffle(lines)

    # train/test 분할
    train_size = int(total_records * train_ratio)
    test_size = total_records - train_size

    train_lines = lines[:train_size]
    test_lines = lines[train_size:]

    print(f"Train 데이터: {len(train_lines)}개 ({len(train_lines) / total_records * 100:.1f}%)")
    print(f"Test 데이터: {len(test_lines)}개 ({len(test_lines) / total_records * 100:.1f}%)")

    # 파일 경로 생성
    base_path = os.path.splitext(input_jsonl_path)[0]
    train_path = f"{base_path}_train.jsonl"
    test_path = f"{base_path}_test.jsonl"

    # 또는 간단하게 train.jsonl, test.jsonl로 저장하고 싶다면:
    # train_path = "train.jsonl"
    # test_path = "test.jsonl"

    # Train 파일 저장
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # Test 파일 저장
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print(f"\n파일 저장 완료!")
    print(f"Train 파일: {train_path}")
    print(f"Test 파일: {test_path}")

    return train_path, test_path


def verify_split(train_path, test_path):
    """
    분할된 파일들의 정보를 확인하는 함수
    """
    print("\n=== 분할 결과 검증 ===")

    # Train 파일 확인
    with open(train_path, 'r', encoding='utf-8') as f:
        train_count = len(f.readlines())

    # Test 파일 확인
    with open(test_path, 'r', encoding='utf-8') as f:
        test_count = len(f.readlines())

    total_count = train_count + test_count

    print(f"Train 파일: {train_count}개")
    print(f"Test 파일: {test_count}개")
    print(f"전체: {total_count}개")
    print(f"Train 비율: {train_count / total_count * 100:.1f}%")
    print(f"Test 비율: {test_count / total_count * 100:.1f}%")

    # 각 파일의 첫 번째 레코드 미리보기
    print(f"\n=== Train 파일 첫 번째 레코드 ===")
    with open(train_path, 'r', encoding='utf-8') as f:
        first_train = json.loads(f.readline().strip())
        for key, value in first_train.items():
            print(f"  {key}: {value}")

    print(f"\n=== Test 파일 첫 번째 레코드 ===")
    with open(test_path, 'r', encoding='utf-8') as f:
        first_test = json.loads(f.readline().strip())
        for key, value in first_test.items():
            print(f"  {key}: {value}")


# 사용 예제
if __name__ == "__main__":
    # JSONL 파일 경로 설정
    input_file = "github_profiles_v5.jsonl"  # 실제 JSONL 파일 경로로 변경

    # 80:20 분할 실행
    train_file, test_file = split_jsonl(input_file, train_ratio=0.8, random_seed=42)

    # 결과 검증
    verify_split(train_file, test_file)