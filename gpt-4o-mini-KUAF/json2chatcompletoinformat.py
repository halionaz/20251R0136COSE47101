import json


def convert_to_chat_format(input_jsonl_path, output_jsonl_path):
    """
    GitHub 프로필 데이터를 GPT fine-tuning용 chat completion format으로 변환하는 함수

    Args:
        input_jsonl_path (str): 입력 JSONL 파일 경로
        output_jsonl_path (str): 출력 JSONL 파일 경로 (chat format)
    """
    print(f"파일 읽는 중: {input_jsonl_path}")

    processed_count = 0
    error_count = 0

    with open(input_jsonl_path, 'r', encoding='utf-8') as input_file, \
            open(output_jsonl_path, 'w', encoding='utf-8') as output_file:

        for line_num, line in enumerate(input_file, 1):
            try:
                # 원본 레코드 파싱
                record = json.loads(line.strip())

                # stack 값 추출 및 제거
                if 'stack' not in record:
                    print(f"경고: {line_num}번째 줄에 'stack' 필드가 없습니다.")
                    error_count += 1
                    continue

                stack_value = record['stack']

                # stack 필드를 제거한 나머지 데이터로 user content 생성
                user_data = {k: v for k, v in record.items() if k != 'stack'}
                user_content = json.dumps(user_data, ensure_ascii=False)

                # assistant content 생성
                assistant_content = f'"stack": "{stack_value}"'

                # Chat completion format으로 변환
                chat_record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ]
                }

                # 출력 파일에 쓰기
                output_file.write(json.dumps(chat_record, ensure_ascii=False) + '\n')
                processed_count += 1

                # 진행상황 출력 (100개마다)
                if processed_count % 100 == 0:
                    print(f"처리 완료: {processed_count}개")

            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 ({line_num}번째 줄): {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"처리 오류 ({line_num}번째 줄): {e}")
                error_count += 1
                continue

    print(f"\n=== 변환 완료 ===")
    print(f"성공적으로 처리된 레코드: {processed_count}개")
    print(f"오류 발생 레코드: {error_count}개")
    print(f"출력 파일: {output_jsonl_path}")

    return processed_count, error_count


def preview_chat_format(chat_jsonl_path, num_samples=2):
    """
    변환된 chat format 파일을 미리보기하는 함수
    """
    print(f"\n=== Chat Format 미리보기 ({num_samples}개 샘플) ===")

    with open(chat_jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            record = json.loads(line.strip())
            print(f"\n--- 샘플 {i + 1} ---")
            print("Messages:")

            for msg in record['messages']:
                role = msg['role']
                content = msg['content']
                print(f"  {role.upper()}:")
                if role == 'user':
                    # User content를 파싱해서 보기 좋게 표시
                    try:
                        user_data = json.loads(content)
                        print(f"    Username: {user_data.get('username', 'N/A')}")
                        print(f"    주요 언어들:")
                        for lang in ['C++', 'C#', 'Python', 'JavaScript', 'Java']:
                            if lang in user_data and user_data[lang] != '0.0%':
                                print(f"      {lang}: {user_data[lang]}")
                        print(f"    Text (처음 100자): {user_data.get('text', '')[:100]}...")
                    except:
                        print(f"    Content: {content[:100]}...")
                else:  # assistant
                    print(f"    Content: {content}")


def validate_chat_format(chat_jsonl_path):
    """
    Chat format이 올바른지 검증하는 함수
    """
    print(f"\n=== Chat Format 검증 ===")

    valid_count = 0
    invalid_count = 0

    with open(chat_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())

                # 필수 구조 확인
                if 'messages' not in record:
                    print(f"오류 ({line_num}): 'messages' 필드 없음")
                    invalid_count += 1
                    continue

                messages = record['messages']
                if len(messages) != 2:
                    print(f"오류 ({line_num}): messages 개수가 2개가 아님 ({len(messages)}개)")
                    invalid_count += 1
                    continue

                # Role 확인
                if messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
                    print(f"오류 ({line_num}): role이 올바르지 않음")
                    invalid_count += 1
                    continue

                # Content 확인
                if not messages[0]['content'] or not messages[1]['content']:
                    print(f"오류 ({line_num}): content가 비어있음")
                    invalid_count += 1
                    continue

                valid_count += 1

            except Exception as e:
                print(f"검증 오류 ({line_num}): {e}")
                invalid_count += 1

    total = valid_count + invalid_count
    print(f"유효한 레코드: {valid_count}개 ({valid_count / total * 100:.1f}%)")
    print(f"유효하지 않은 레코드: {invalid_count}개 ({invalid_count / total * 100:.1f}%)")

    return valid_count, invalid_count


# 사용 예제
if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "github_profiles_v5_train.jsonl"  # 실제 파일 경로로 변경
    output_file = "github_profiles_v5_train_chat_format.jsonl"

    # Chat format으로 변환
    processed, errors = convert_to_chat_format(input_file, output_file)

    if processed > 0:
        # 결과 미리보기
        preview_chat_format(output_file)

        # Format 검증
        validate_chat_format(output_file)