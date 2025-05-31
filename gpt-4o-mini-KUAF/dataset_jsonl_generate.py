import pandas as pd
import json


def csv_to_jsonl(csv_file_path, output_jsonl_path):
    """
    CSV 파일을 JSONL 형식으로 변환하는 함수 (인코딩 자동 처리)

    Args:
        csv_file_path (str): 입력 CSV 파일 경로
        output_jsonl_path (str): 출력 JSONL 파일 경로
    """
    # 여러 인코딩을 순서대로 시도
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'utf-8-sig']
    df = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
            print(f"성공적으로 읽음: {encoding} 인코딩 사용")
            break
        except UnicodeDecodeError:
            print(f"{encoding} 인코딩 실패, 다른 인코딩 시도...")
            continue
        except Exception as e:
            print(f"{encoding} 인코딩에서 오류 발생: {e}")
            continue

    if df is None:
        raise ValueError("모든 인코딩 시도 실패. CSV 파일을 읽을 수 없습니다.")

    # 언어 컬럼들 (Assembly부터 TypeScript까지)
    language_columns = ['Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java',
                        'JavaScript', 'Kotlin', 'MATLAB', 'PHP', 'Python',
                        'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript']

    # JSONL 파일로 변환
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 각 행을 딕셔너리로 변환
            record = {}

            # username 추가
            record['username'] = row['username']

            # 언어별 사용 비율 추가 (퍼센트로 변환)
            for lang in language_columns:
                percentage = row[lang] * 100 if pd.notna(row[lang]) else 0
                record[f'{lang}'] = f"{percentage:.1f}%"

            # text (repo description) 추가
            record['text'] = row['text'] if pd.notna(row['text']) else ""

            # stack (기술 스택) 추가
            record['stack'] = row['stack'] if pd.notna(row['stack']) else ""

            # JSON으로 변환하여 파일에 쓰기
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"변환 완료! {len(df)}개의 레코드가 {output_jsonl_path}에 저장되었습니다.")


def preview_jsonl(jsonl_path, num_records=3):
    """
    JSONL 파일의 처음 몇 개 레코드를 미리보기하는 함수
    """
    print(f"\n{jsonl_path} 파일 미리보기:")
    print("-" * 80)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_records:
                break
            record = json.loads(line.strip())
            print(f"Record {i + 1}:")
            for key, value in record.items():
                print(f"  {key}: {value}")
            print()


# 사용 예제
if __name__ == "__main__":
    # CSV 파일 경로와 출력 JSONL 파일 경로 설정
    input_csv = "github_profiles_total_v5.csv"  # 실제 CSV 파일 경로로 변경
    output_jsonl = "processed_dataset.jsonl"

    # CSV를 JSONL로 변환
    csv_to_jsonl(input_csv, output_jsonl)

    # 결과 미리보기
    preview_jsonl(output_jsonl)