import json


def remove_stack_from_jsonl(input_jsonl_path, output_jsonl_path):
    """
    JSONL 파일에서 'stack' 필드를 제거하는 함수

    Args:
        input_jsonl_path (str): 입력 JSONL 파일 경로 (test.jsonl)
        output_jsonl_path (str): 출력 JSONL 파일 경로 (test_without_stack.jsonl)
    """
    print(f"파일 읽는 중: {input_jsonl_path}")

    processed_count = 0

    with open(input_jsonl_path, 'r', encoding='utf-8') as input_file, \
            open(output_jsonl_path, 'w', encoding='utf-8') as output_file:

        for line_num, line in enumerate(input_file, 1):
            try:
                # JSON 파싱
                record = json.loads(line.strip())

                # 'stack' 필드 제거 (있는 경우에만)
                if 'stack' in record:
                    del record['stack']

                # 수정된 레코드를 출력 파일에 쓰기
                output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"경고: {line_num}번째 줄에서 JSON 파싱 오류: {e}")
                continue
            except Exception as e:
                print(f"경고: {line_num}번째 줄에서 오류 발생: {e}")
                continue

    print(f"처리 완료! {processed_count}개의 레코드가 {output_jsonl_path}에 저장되었습니다.")
    return processed_count


def compare_files(original_path, modified_path, num_samples=3):
    """
    원본 파일과 수정된 파일을 비교하여 결과를 확인하는 함수
    """
    print(f"\n=== 파일 비교 ({num_samples}개 샘플) ===")

    with open(original_path, 'r', encoding='utf-8') as orig_file, \
            open(modified_path, 'r', encoding='utf-8') as mod_file:

        for i in range(num_samples):
            print(f"\n--- 레코드 {i + 1} ---")

            # 원본 파일 레코드
            orig_line = orig_file.readline().strip()
            if not orig_line:
                break
            orig_record = json.loads(orig_line)

            # 수정된 파일 레코드
            mod_line = mod_file.readline().strip()
            if not mod_line:
                break
            mod_record = json.loads(mod_line)

            print("원본 파일 키들:", list(orig_record.keys()))
            print("수정된 파일 키들:", list(mod_record.keys()))

            # stack 필드 확인
            if 'stack' in orig_record:
                print(f"제거된 stack 값: {orig_record['stack']}")
            else:
                print("원본에 stack 필드가 없었음")

            if 'stack' in mod_record:
                print("⚠️ 경고: 수정된 파일에 여전히 stack 필드가 존재함")
            else:
                print("✅ stack 필드가 성공적으로 제거됨")


def verify_file_counts(original_path, modified_path):
    """
    두 파일의 레코드 수가 동일한지 확인하는 함수
    """
    print(f"\n=== 레코드 수 확인 ===")

    with open(original_path, 'r', encoding='utf-8') as f:
        original_count = len(f.readlines())

    with open(modified_path, 'r', encoding='utf-8') as f:
        modified_count = len(f.readlines())

    print(f"원본 파일 레코드 수: {original_count}")
    print(f"수정된 파일 레코드 수: {modified_count}")

    if original_count == modified_count:
        print("✅ 레코드 수가 일치합니다.")
    else:
        print("⚠️ 경고: 레코드 수가 일치하지 않습니다!")


# 사용 예제
if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "github_profiles_v5_test.jsonl"  # 또는 실제 test 파일 경로
    output_file = "github_profiles_v5_test_without_stack.jsonl"

    # stack 필드 제거 실행
    processed_count = remove_stack_from_jsonl(input_file, output_file)

    # 결과 검증
    if processed_count > 0:
        compare_files(input_file, output_file)
        verify_file_counts(input_file, output_file)