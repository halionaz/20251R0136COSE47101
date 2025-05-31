import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
def load_processed_data():
    """전처리된 데이터 로드"""
    print("📁 전처리된 데이터 로드 중...")

    # SMOTE 적용된 분할 데이터 사용
    X_train = np.load('X_train_balanced.npy')
    X_test = np.load('X_test_balanced.npy')
    y_train = np.load('y_train_balanced.npy')
    y_test = np.load('y_test_balanced.npy')

    # 클래스 가중치 로드
    with open('class_weights.pkl', 'rb') as f:
        class_weights = pickle.load(f)



   # 타깃 스택 고정 (7개)
    target_stacks = ['Server', 'System', 'Visualization', 'Frontend', 'Android', 'ML-Data', 'iOS']

    print(f"✅ 데이터 로드 완료:")
    print(f"   훈련 데이터: {X_train.shape}")
    print(f"   테스트 데이터: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test shape: {y_test.shape}")

    # ⚠️ y 데이터가 25차원이라면 7차원으로 축소 필요
    if y_train.shape[1] != 7:
        print(f"⚠️ y 데이터 차원 불일치: {y_train.shape[1]} != 7")
        print("   → 메타데이터에서 올바른 타깃 스택 순서 확인 필요")

        # 메타데이터에서 실제 스택 순서 확인
        with open('metadata_enhanced.pkl', 'rb') as f:
            metadata = pickle.load(f)

        original_target_stacks = metadata['target_stacks']
        print(f"   원본 타깃 스택: {original_target_stacks}")

        # 7개 기본 스택의 인덱스 찾기
        target_indices = []
        for stack in target_stacks:
            if stack in original_target_stacks:
                target_indices.append(original_target_stacks.index(stack))
            else:
                print(f"❌ {stack}이 원본 데이터에 없습니다!")

        print(f"   사용할 인덱스: {target_indices}")

        # y 데이터를 7차원으로 축소
        y_train = y_train[:, target_indices]
        y_test = y_test[:, target_indices]

        print(f"✅ y 데이터 차원 수정: {y_train.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test, class_weights, target_stacks

# 개선된 모델 정의
def create_enhanced_model(input_dim, output_dim):
    """기존 모델을 기반으로 한 개선된 모델"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        # 첫 번째 블록
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # 두 번째 블록
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # 세 번째 블록
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # 출력층
        layers.Dense(output_dim, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model

# 멀티라벨 평가 함수
def evaluate_multilabel_model(model, X_test, y_test, target_stacks, threshold=0.2):
    """멀티라벨 모델 평가"""
    print(f"\n📊 모델 평가 중...")
    print(f"   타깃 스택: {target_stacks}")
    print(f"   y_test shape: {y_test.shape}")

    # 각 스택별 실제 분포 확인
    print(f"\n📋 테스트 세트 스택별 분포:")
    for i, stack in enumerate(target_stacks):
        count = np.sum(y_test[:, i])
        percentage = (count / len(y_test)) * 100
        print(f"   {stack}: {count}개 ({percentage:.1f}%)")

    # 예측
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > threshold).astype(int)
    # 모델 학습 코드의 evaluate 함수 전에 추가
    print("🔍 데이터 구조 디버깅:")
    print(f"y_test 샘플 5개:")
    for i in range(5):
        active_labels = np.where(y_test[i] == 1)[0]
        stack_names = [target_stacks[j] for j in active_labels]
        print(f"  샘플 {i}: {stack_names}")

    print(f"\ny_pred_prob 샘플 5개:")
    for i in range(5):
        probs = y_pred_prob[i]
        top_indices = np.argsort(probs)[::-1][:3]
        top_stacks = [(target_stacks[j], probs[j]) for j in top_indices]
        print(f"  샘플 {i}: {top_stacks}")
    # 전체 정확도 (모든 라벨이 정확해야 함)
    exact_match_ratio = np.mean(np.all(y_pred == y_test, axis=1))

    # Hamming Loss (라벨별 평균 오류율)
    hamming_loss_score = hamming_loss(y_test, y_pred)

    # Jaccard Score (IoU)
    jaccard_score_macro = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
    jaccard_score_micro = jaccard_score(y_test, y_pred, average='micro', zero_division=0)

    print(f"🎯 멀티라벨 성능 지표:")
    print(f"   정확한 매칭 비율: {exact_match_ratio:.4f} ({exact_match_ratio:.2%})")
    print(f"   Hamming Loss: {hamming_loss_score:.4f}")
    print(f"   Jaccard Score (Macro): {jaccard_score_macro:.4f}")
    print(f"   Jaccard Score (Micro): {jaccard_score_micro:.4f}")

    # 라벨별 성능
    print(f"\n📋 스택별 성능:")
    for i, stack in enumerate(target_stacks):
        true_positive = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1))
        false_positive = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1))
        false_negative = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"   {stack}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return exact_match_ratio, hamming_loss_score, jaccard_score_macro

# Top-k 정확도 계산
def calculate_topk_accuracy(model, X_test, y_test, target_stacks, k_values=[1, 2, 3]):
    """Top-k 정확도 계산 (기존 방식과 호환)"""
    print(f"\n🎯 Top-k 정확도 계산 중...")

    y_pred_prob = model.predict(X_test, verbose=0)

    results = {}
    for k in k_values:
        # 각 샘플에 대해 상위 k개 예측
        top_k_preds = np.argsort(y_pred_prob, axis=1)[:, -k:]

        # 실제 라벨과 비교
        matches = []
        for i in range(len(y_test)):
            true_labels = set(np.where(y_test[i] == 1)[0])
            pred_labels = set(top_k_preds[i])
            # 교집합이 있으면 성공
            matches.append(len(true_labels & pred_labels) > 0)

        accuracy = np.mean(matches)
        results[k] = accuracy
        print(f"   Top-{k} 정확도: {accuracy:.4f} ({accuracy:.2%})")

    return results

# 학습 및 평가 함수
def train_and_evaluate():
    """전체 학습 및 평가 프로세스"""

    # 1. 데이터 로드
    X_train, X_test, y_train, y_test, class_weights, target_stacks = load_processed_data()

    # 2. 모델 생성
    input_dim = X_train.shape[1]
    output_dim = 7

    print(f"\n🤖 모델 생성 중...")
    print(f"   입력 차원: {input_dim}")
    print(f"   출력 차원: {output_dim} (7개 기본 스택)")

    model = create_enhanced_model(input_dim, output_dim)
    model.summary()

    # 3. 콜백 설정
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_multilabel_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 4. 모델 학습
    print(f"\n🚀 모델 학습 시작...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=100,
        class_weight=class_weights,  # 클래스 가중치 적용
        callbacks=callbacks_list,
        verbose=1
    )
    # 5. 모델 평가
    print(f"\n📊 최종 평가:")

    # 멀티라벨 평가
    exact_match, hamming_loss, jaccard_macro = evaluate_multilabel_model(
        model, X_test, y_test, target_stacks
    )

    # Top-k 정확도 (기존 방식과 비교용)
    topk_results = calculate_topk_accuracy(model, X_test, y_test, target_stacks)

    # 6. 결과 시각화
    plot_training_history(history)
    plot_prediction_analysis(model, X_test, y_test, target_stacks)

    # 7. 최종 결과 요약
    print(f"\n🎉 최종 성능 요약:")
    print(f"   정확한 매칭: {exact_match:.2%}")
    print(f"   Top-1 정확도: {topk_results[1]:.2%}")
    print(f"   Top-2 정확도: {topk_results[2]:.2%}")
    print(f"   Jaccard Score: {jaccard_macro:.4f}")

    return model, history, exact_match, topk_results

# 학습 곡선 시각화
def plot_training_history(history):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Accuracy
    axes[0,1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Precision
    if 'precision' in history.history:
        axes[1,0].plot(history.history['precision'], label='Training Precision')
        axes[1,0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

    # Recall
    if 'recall' in history.history:
        axes[1,1].plot(history.history['recall'], label='Training Recall')
        axes[1,1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 예측 분석 시각화
def plot_prediction_analysis(model, X_test, y_test, target_stacks):
    """예측 분석 시각화"""
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 실제 vs 예측 분포
    true_counts = np.sum(y_test, axis=0)
    pred_counts = np.sum(y_pred, axis=0)

    x = range(len(target_stacks))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], true_counts, width, label='실제', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], pred_counts, width, label='예측', alpha=0.8)
    axes[0].set_xlabel('스택')
    axes[0].set_ylabel('샘플 수')
    axes[0].set_title('실제 vs 예측 분포')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(target_stacks, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 예측 확률 분포
    axes[1].boxplot([y_pred_prob[:, i] for i in range(len(target_stacks))],
                    labels=target_stacks)
    axes[1].set_xlabel('스택')
    axes[1].set_ylabel('예측 확률')
    axes[1].set_title('스택별 예측 확률 분포')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 메인 실행
if __name__ == "__main__":
    print("🚀 멀티라벨 스택 예측 모델 학습 시작!")
    print("=" * 60)

    # 학습 및 평가 실행
    model, history, exact_match, topk_results = train_and_evaluate()

    print(f"\n✅ 학습 완료!")
    print(f"📁 저장된 파일: best_multilabel_model.keras")