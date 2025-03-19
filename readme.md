# 음악 장르 분류 프로젝트

## 프로젝트 개요
이 프로젝트는 음악 특성 데이터를 활용하여 머신러닝 모델로 음악 장르를 분류하는 시스템을 구현합니다. 다양한 머신러닝 알고리즘을 비교 평가하고, K-fold 교차 검증 및 그리드서치를 통한 하이퍼파라미터 최적화를 수행합니다. 자세한 프로젝트 진행 내용은 링크를 참고해주세요.

[국립공주대학교 창의적 문제해결 개인 텀 프로젝트 음악 장르 분류](https://www.notion.so/183bf0122d4081bfa884d89d153a5b72#1b9bf0122d40808f86dce44410a076eb)

## 모델 성능 결과
| Test Set  | DecisionTree (전처리 전) | DecisionTree (전처리 후) | XGB(roc_auc 튜닝) | MLP(roc_auc 튜닝) | stacking vote(meta: SVM) |
|-----------|----------|-------------------------|----------------------|---------------------|-----------------------------------|
| mean accuracy  | 0.5703   | 0.6972  | 0.7875  | 0.7698  | 0.7917  |
| mean precision | 0.5699   | 0.6967  | 0.7850  | 0.7686  | 0.7903  |
| mean recall    | 0.5703   | 0.6972  | 0.7875  | 0.7698  | 0.7917  |
| mean f1 score  | 0.5696   | 0.6967  | 0.7856  | 0.7686  | 0.7898  |
| auc           | 0.7708   | 0.8376  | 0.9791  | 0.9759  | 0.9679  |

1. 전처리 효과
    - 전처리 후 Decision Tree 모델은 전처리 전보다 모든 평가 지표에서 약 12%정도 향상되었습니다.
2. 모델 비교
    - XGB와 스태킹(meta:SVM) 모델이 가장 우수한 성능을 보입니다.
    - 스태킹 모델은 accuracy(0.7917)와 recall(0.7917) 면에서 가장 높은 수치를 보여줍니다.
    - XGB는 AUC(0.9791) 측면에서 가장 뛰어나며, 전반적으로 모든 지표에서 높은 성능을 보입니다.
3. 복잡한 모델
    - 단순한 Decision Tree 보다 복잡한 모델의 성능이 전반적으로 10-20% 더 높은 성능을 보이며 AUC 지표에서 복잡한 모델이 매우 높은 수치를 기록하여 분류 능력이 좋았습니다.
4. 스태킹 효과
    - meta 분류기로 SVM을 사용한 스태킹 기법이 accuracy와 recall 측면에서 가장 우수하여, 여러 모델의 장점을 결합했을 때의 시너지 효과를 보여줍니다.
    - 다만 AUC에서는 XGB보다 약간 낮은 성능(0.9679)을 보입니다.

## 데이터
- 데이터셋: genres_v2.csv (프로젝트 루트 디렉토리의 data 폴더에 위치)
- 특성: 해당 데이터셋은 음악 트랙의 다양한 특성을 포함하며, 일부 불필요한 특성은 전처리 과정에서 제거됩니다.
- 타겟: 'genre' 컬럼 (음악 장르)
- 참고: 'Underground Rap'과 'Pop' 클래스는 분석에서 제외됩니다.

## 필요 라이브러리
```
numpy
pandas
scikit-learn
xgboost
lightgbm
scipy
matplotlib
seaborn
```

## 프로젝트 구조
```

data/
└── genres_v2.csv    # 데이터셋
ipynb/
    ├── 0_baseline.ipynb        # Decision 모델의 baseline 코드
    ├── 1_eda.ipynb             # 데이터 eda 결과
    ├── 2_outlier.ipynb         # 데이터 outlier 실험
    ├── 3_data_imbalance.ipynb  # imbalnce 해결 실험
    ├── 4_scaler.ipynb          # scaler 실험
    └── extra_class.ipynb       # 번외로 클래스 제거 및 통합 실험
src/  
    ├── /logs                   # 모델 학습 결과 log
    ├── main.py                 # 메인 실행 파일
    ├── logger.py               # 로깅 기능
    ├── config.py               # 모델 및 하이퍼파라미터 설정 관리
    ├── data_processor.py       # 데이터 로드 및 전처리 기능
    ├── model_factory.py        # 다양한 모델 생성
    ├── model_trainer.py        # 모델 학습 및 평가
    ├── visualizer.py           # 결과 시각화
    ├── grid_search.py          # 그리드서치를 통한 하이퍼파라미터 최적화
    ├── ensemble_model.py       # 앙상블 모델 생성 클래스
```

## 사용 방법

### 1. 단일 모델 실행
```bash
python main.py --model xgb
```

### 2. 그리드서치를 이용한 하이퍼파라미터 최적화
```bash
python main.py --model xgb --grid_search
```

### 3. 모델별 하이퍼파라미터 설정
```bash
# XGBoost 모델 사용 예시
python main.py --model xgb --xgb_n_estimators 150 --xgb_max_depth 5

# MLP 모델 사용 예시
python main.py --model mlp --mlp_hidden_layers "(100, 50)" --mlp_max_iter 500
```

### 4. 앙상블 모델 실행
```bash
# 보팅 앙상블 (기본)
python main.py --ensemble --ensemble_type voting --voting_type soft --ensemble_models xgb gb

# 스태킹 앙상블
python main.py --ensemble --ensemble_type stacking --ensemble_models xgb svm --meta_model lr
```

## 지원되는 모델
### 단일 모델
- Decision Tree (dt)
- Random Forest (rf)
- Gradient Boosting (gb)
- XGBoost (xgb)
- Support Vector Machine (svm)
- Multi-layer Perceptron (mlp)

### 앙상블 모델
- 보팅 앙상블 (Voting Ensemble): 여러 모델의 예측을 투표 방식으로 결합
  - Hard Voting: 다수결 원칙에 따라 결정
  - Soft Voting: 각 모델의 확률을 평균하여 결정
- 스태킹 앙상블 (Stacking Ensemble): 여러 모델의 예측을 메타 모델의 입력으로 사용

## 주요 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|-----|-------|
| --model | 사용할 모델 | dt |
| --k | K-fold 교차 검증 폴드 수 | 5 |
| **--grid_search** | 그리드서치 사용 여부 | False |
| --scoring | 그리드서치 평가 지표 | f1_weighted, roc |
| **--ensemble** | 앙상블 실행 여부 | False |
| --ensemble_type | 앙상블 유형 (voting, stacking) | voting |
| --voting_type | 보팅 방식 (hard 또는 soft) | soft |
| --ensemble_models | 앙상블에 포함할 모델 목록 | xgb, gb |
| --meta_model | 스태킹 앙상블의 메타 모델 | lr |
| --n_jobs | 병렬 처리 작업 수 | -1 (모든 코어) |


## 결과 분석
학습 과정에서 다음과 같은 결과물이 생성됩니다:
- logs/ 디렉토리에 모델별 학습 로그
- 각 모델의 혼동 행렬(Confusion Matrix) 시각화
- 그리드서치 사용 시 파라미터별 성능 그래프

## 구현 세부사항

### 데이터 전처리
- 불필요한 특성 제거
- 특정 장르 클래스 제외
- 이상치 처리 (Z-score 3 기준)
- QuantileTransformer를 사용한 정규 분포 변환

### 모델 학습 및 평가
- StratifiedKFold를 사용한 교차 검증
- 정확도, 정밀도, 재현율, F1 점수, AUC 등 다양한 평가 지표 사용
- 학습 및 검증 과정의 상세 로깅

### 하이퍼파라미터 최적화
- GridSearchCV를 통한 최적 파라미터 탐색(scoring: roc, f1)
- 파라미터별 중요도 시각화

### 앙상블 기법
- **보팅 앙상블**: scikit-learn의 VotingClassifier를 사용하여 여러 분류기의 예측을 결합
- **스태킹 앙상블**: 기본 모델의 예측을 메타 모델의 입력으로 사용하는 계층적 접근법