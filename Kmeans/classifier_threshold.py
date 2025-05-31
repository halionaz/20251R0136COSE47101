import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, hamming_loss, jaccard_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(pkl_path):
    print("📂 데이터 로딩 중...")
    df = pd.read_pickle(pkl_path)
    # 컬럼 분류
    basic_cols = [col for col in df.columns if not col.startswith('bert_')]
    bert_name_cols = [col for col in df.columns if col.startswith('bert_name_')]
    bert_desc_cols = [col for col in df.columns if col.startswith('bert_desc_')]
    
    # 타겟 변수 확인 (stack_list 사용)
    if 'stack_list' in df.columns:
        print(f"\n🎯 타겟 변수 (stack_list) 분석:")
        
        # 스택 리스트 분석
        all_stacks = []
        valid_stack_lists = []
        
        for stack_list in df['stack_list']:
            if isinstance(stack_list, list) and len(stack_list) > 0:
                all_stacks.extend(stack_list)
                valid_stack_lists.append(stack_list)
        
        stack_counts = Counter(all_stacks)
        print(f"총 고유 스택 수: {len(stack_counts)}") #7
        return df, basic_cols, bert_name_cols, bert_desc_cols, stack_counts, valid_stack_lists

def prepare_features(df, basic_cols, bert_name_cols, bert_desc_cols):
    """
    모델 학습을 위한 피처 준비 + BERT 임베딩 PCA 적용
    """
    print("\n🔧 피처 준비 중...")
    exclude_cols = {'user_ID', 'username', 'repo_names', 'description', 'stack', 'stack_list', 'note', 'repo_count'}
    language_cols = [col for col in basic_cols if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    print(f"🎯 언어 통계 피처: {language_cols}")

    basic_info_cols = ['repo_count'] if 'repo_count' in df.columns else []
    print("📉 PCA 적용 중 (BERT name/desc)...")
    
    # PCA 적용 (차원 수를 적당히 줄임)
    pca_name = PCA(n_components=0.9, random_state=42)
    pca_desc = PCA(n_components=0.9, random_state=42)

    bert_name_pca = pca_name.fit_transform(df[bert_name_cols])
    bert_desc_pca = pca_desc.fit_transform(df[bert_desc_cols])

    bert_name_df = pd.DataFrame(bert_name_pca, columns=[f'pca_name_{i}' for i in range(bert_name_pca.shape[1])])
    bert_desc_df = pd.DataFrame(bert_desc_pca, columns=[f'pca_desc_{i}' for i in range(bert_desc_pca.shape[1])])

    df_pca = pd.concat([df.reset_index(drop=True), bert_name_df, bert_desc_df], axis=1)
    feature_columns = language_cols + basic_info_cols+list(bert_name_df.columns) + list(bert_desc_df.columns)
    X = df_pca[feature_columns].copy()

    print(f"📊 총 피처 수 (PCA 적용 후): {len(feature_columns)}")
    print(f"• 언어 통계: {len(language_cols)}개")
    print(f"• 기본 정보: {len(basic_info_cols)}개")
    print(f"• BERT PCA name: {bert_name_pca.shape[1]}개")
    print(f"• BERT PCA desc: {bert_desc_pca.shape[1]}개")

    return X, feature_columns, language_cols, pca_name, pca_desc

def prepare_target_onehot(df, min_samples=10):
    """
    스택을 원핫인코딩으로 변환하는 함수
    """
    print(f"\n🎯 스택 원핫인코딩 준비 중 (최소 {min_samples}개 샘플)...")
    
    # 유효한 stack_list만 선택
    valid_mask = df['stack_list'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    print(f"유효한 타겟 데이터: {valid_mask.sum()}개")
    
    # 스택별 빈도 계산
    all_stacks = []
    for stack_list in df.loc[valid_mask, 'stack_list']:
        all_stacks.extend(stack_list)
    
    stack_counts = Counter(all_stacks)
    frequent_stacks = [stack for stack, count in stack_counts.items() if count >= min_samples]
    print(f"최소 {min_samples}개 샘플을 가진 스택 수: {len(frequent_stacks)}")
    
    
    # 빈번한 스택만 포함하는 스택 리스트 필터링
    filtered_stack_lists = []
    filtered_indices = []
    
    for idx, stack_list in df.loc[valid_mask, 'stack_list'].items():
        filtered_list = [stack for stack in stack_list if stack in frequent_stacks]
        if filtered_list:  # 빈 리스트가 아닌 경우만 포함
            filtered_stack_lists.append(filtered_list)
            filtered_indices.append(idx)
    
    print(f"필터링 후 데이터: {len(filtered_indices)}개") #
    
    # MultiLabelBinarizer로 원핫인코딩
    mlb = MultiLabelBinarizer()
    y_onehot = mlb.fit_transform(filtered_stack_lists)
    
    # 유효한 마스크 생성 노이즈 제거. 잘 등장하지 않는 스택 제거거
    final_valid_mask = pd.Series(False, index=df.index)
    final_valid_mask.iloc[filtered_indices] = True
    print(f"✅ 원핫인코딩 완료")
    print(f"타겟 shape: {y_onehot.shape}")
    print(f"스택 클래스: {list(mlb.classes_)}")
    # 클래스별 샘플 수 확인
    class_counts = y_onehot.sum(axis=0)
    print(f"\n클래스별 샘플 수:")
    for i, (class_name, count) in enumerate(zip(mlb.classes_, class_counts)):
        print(f"  {class_name}: {count}개")
    return y_onehot, final_valid_mask, mlb, frequent_stacks

def train_multilabel_models(X_train, y_train, X_test, y_test, mlb):
    """
    멀티라벨 분류 모델 학습 및 평가
    """
    print("\n🤖 멀티라벨 모델 학습 중...")
    
    base_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    }
    
    results = {}
    
    for name, base_model in base_models.items():
        print(f"\n🔄 {name} (멀티라벨) 학습 중...")
        
        '''# Gradient Boosting 튜닝
        if name == 'Gradient Boosting':
            param_grid = {
                'estimator__n_estimators': [100, 200],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__max_depth': [3, 5],
                'estimator__subsample': [0.8, 1.0],
                
            }
            model = GridSearchCV(
                estimator=MultiOutputClassifier(base_model, n_jobs=-1),
                param_grid=param_grid,
                scoring='f1_micro',
                cv=3,
                n_jobs=-1,
                verbose=1
            )

        # XGBoost 튜닝
        elif name == 'XGBoost':
            param_grid = {
                'estimator__n_estimators': [100, 200],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__max_depth': [3, 5],
                'estimator__subsample': [0.8, 1.0],
                'estimator__colsample_bytree': [0.8, 1.0],
                
                }
            model = GridSearchCV(
                estimator=MultiOutputClassifier(base_model, n_jobs=-1),
                param_grid=param_grid,
                scoring='f1_micro',
                cv=3,
                
                n_jobs=-1,
                verbose=1
            )

        # 나머지 모델은 기본 MultiOutputClassifier 사용
        else:'''
        model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        # 모델 학습
        model.fit(X_train, y_train)
        if hasattr(model, "best_params_"):
            print(f"📌 {name} 최적 파라미터: {model.best_params_}")
        # 예측 (확률)
        y_pred_proba = model.predict_proba(X_test)
        
        # 각 클래스별로 최고 확률을 가진 예측을 선택
        y_pred_binary = np.zeros_like(y_test)
        
        # 방법 1: 각 샘플마다 가장 높은 확률을 가진 클래스를 1로 설정
        for i in range(len(y_test)):
            if len(y_pred_proba) > 0:
                sample_probs = []
                for j in range(y_train.shape[1]):  # 각 클래스에 대해
                    if len(y_pred_proba[j][i]) > 1:  # 이진 분류의 경우
                        sample_probs.append(y_pred_proba[j][i][1])  # 양성 클래스 확률
                    else:
                        sample_probs.append(y_pred_proba[j][i][0])
                
                # 상위 확률을 가진 클래스들을 선택 (임계값 기반)
                threshold = 0.6  # 조정 가능한 임계값
                for j, prob in enumerate(sample_probs):
                    if prob > threshold:
                        y_pred_binary[i, j] = 1
                
                # 최소한 하나의 클래스는 예측되도록 보장
                if y_pred_binary[i].sum() == 0:
                    best_class = np.argmax(sample_probs)
                    y_pred_binary[i, best_class] = 1
        
        # 평가 메트릭 계산
        hamming = hamming_loss(y_test, y_pred_binary)
        jaccard = jaccard_score(y_test, y_pred_binary, average='samples', zero_division=0)
        
        # 정확한 매치 비율 (모든 라벨이 정확히 일치하는 비율)
        exact_match = np.mean(np.all(y_test == y_pred_binary, axis=1))
        
        # 커스텀 평가: 예측과 실제가 겹치는 것이 있으면 성공
        #flexible_accuracy = calculate_flexible_accuracy(y_test, y_pred_binary)
        
        results[name] = {
            'model': model,
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'exact_match': exact_match,
            #'flexible_accuracy': flexible_accuracy,
            'predictions': y_pred_binary,
            'predictions_proba': y_pred_proba
        }
        
        print(f"✅ {name} 완료!")
        print(f"   Hamming Loss: {hamming:.4f}")
        print(f"   Jaccard Score: {jaccard:.4f}")
        print(f"   Exact Match: {exact_match:.4f}")
        #print(f"   Flexible Accuracy: {flexible_accuracy:.4f}")
    
    return results

'''def calculate_flexible_accuracy(y_true, y_pred):
    """
    예측된 스택 중 하나라도 실제 스택과 일치하면 정답으로 처리하는 정확도
    """
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        # 실제 스택 (1인 위치)
        true_stacks = set(np.where(y_true[i] == 1)[0])
        # 예측된 스택 (1인 위치)
        pred_stacks = set(np.where(y_pred[i] == 1)[0])
        
        # 교집합이 있으면 성공
        if len(true_stacks.intersection(pred_stacks)) > 0:
            correct += 1
    
    return correct / total'''

def evaluate_model_performance(results, y_test, mlb):
    """
    모델 성능 평가 및 출력
    """
    print("\n🏆 모델 성능 비교:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Hamming':<8} | {'Jaccard':<8} | {'Exact':<8}")
    print("-" * 80)
    
    best_model_name = None
    best_score = 0
    
    for name, result in results.items():
        # flexible_acc = result['flexible_accuracy']
        exact_match = result['exact_match']
        print(f"{name:<20} | {result['hamming_loss']:<8.4f} | {result['jaccard_score']:<8.4f} | {result['exact_match']:<8.4f}")
        
        # Flexible Accuracy를 기준으로 최고 모델 선택
        if exact_match > best_score:
            best_score = exact_match
            best_model_name = name
    
    print("-" * 80)
    print(f"🥇 최고 성능 모델: {best_model_name} (Exact_match: {best_score:.4f})")
    
    # 상세 분석
    best_result = results[best_model_name]
    y_pred = best_result['predictions']
    
    print(f"\n📊 {best_model_name} 상세 분석:")
    
    # 클래스별 성능 분석
    print("\n클래스별 성능:")
    for i, class_name in enumerate(mlb.classes_):
        true_positive = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1))
        false_positive = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1))
        false_negative = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {class_name:<15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return best_result['model'], best_model_name, best_result

def visualize_results(results, mlb):
    """
    결과 시각화
    """
    print("\n📈 결과 시각화 중...")
    
    # 성능 비교 차트
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    model_names = list(results.keys())
    metrics = {
        'Hamming Loss': [results[name]['hamming_loss'] for name in model_names],
        'Jaccard Score': [results[name]['jaccard_score'] for name in model_names],
        'Exact Match': [results[name]['exact_match'] for name in model_names],
        # 'Flexible Accuracy': [results[name]['flexible_accuracy'] for name in model_names]
    }
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        row = idx // 2
        col = idx % 2
        
        bars = axes[row, col].bar(model_names, values, color=colors[idx], alpha=0.7)
        axes[row, col].set_title(f'{metric_name}')
        axes[row, col].set_ylabel('Score')
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    # 파일 경로 설정
    pkl_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/pkl_data/github_profiles_with_bert_processed_v2.pkl'
    
    try:
        # 1. 데이터 로드 및 탐색
        df, basic_cols, bert_name_cols, bert_desc_cols, stack_counts, valid_stack_lists = load_and_explore_data(pkl_path)
        
        if stack_counts is None:
            return
        
        # 2. 피처 준비
        X, feature_columns, language_cols, pca_name, pca_desc = prepare_features(df, basic_cols, bert_name_cols, bert_desc_cols)
        
        # 3. 타겟 준비 - 원핫인코딩 방식
        print("\n🎯 원핫인코딩 방식 선택")
        y, valid_mask, mlb, frequent_stacks = prepare_target_onehot(df, min_samples=8)
        
        # 유효한 데이터만 선택
        X = X.loc[valid_mask].reset_index(drop=True)
        
        print(f"\n📊 최종 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {y.shape}")
        
        # 4. 데이터 분할
        print("\n🔀 데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"학습 데이터: {X_train.shape}, {y_train.shape}")
        print(f"테스트 데이터: {X_test.shape}, {y_test.shape}")
        
        # 5. 피처 스케일링
        print("\n⚖️ 피처 스케일링 중...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. 멀티라벨 모델 학습 및 평가
        results = train_multilabel_models(X_train_scaled, y_train, X_test_scaled, y_test, mlb)
        
        # 7. 모델 성능 평가
        best_model, best_model_name, best_result = evaluate_model_performance(results, y_test, mlb)
        
        # 8. 결과 시각화
        visualize_results(results, mlb)
        
        # print(f"Flexible Accuracy: {best_result['flexible_accuracy']:.4f}")
        print(f"Exact Match: {best_result['exact_match']:.4f}")
        
        print("\n🎉 스택 분류 모델 학습 완료!")
        
        # 10. 예시 예측 출력
        print("\n🔍 예측 예시 (처음 5개 샘플):")
        y_pred = best_result['predictions']
        for i in range(min(5, len(y_test))):
            true_stacks = [mlb.classes_[j] for j in range(len(mlb.classes_)) if y_test[i, j] == 1]
            pred_stacks = [mlb.classes_[j] for j in range(len(mlb.classes_)) if y_pred[i, j] == 1]
            
            print(f"샘플 {i+1}:")
            print(f"  실제: {true_stacks}")
            print(f"  예측: {pred_stacks}")
            print(f"  일치: {'✅' if len(set(true_stacks).intersection(set(pred_stacks))) > 0 else '❌'}")
            print()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()