import argparse
from copy import deepcopy

def parse_arguments():
    parser = argparse.ArgumentParser(description='음악 장르 분류 모델 학습')
    
    parser.add_argument('--model', type=str, default='dt', 
                        choices=['dt', 'rf', 'gb', 'xgb', 'svm', 'mlp'],
                        help='사용할 모델 dt, rf, gb, xg, svm, mlp')
    
    parser.add_argument('--k', type=int, default=5, 
                        help='K-fold 교차 검증에 사용할 폴드 수')
    
    # 그리드서치 관련 파라미터
    parser.add_argument('--grid_search', action='store_true',
                        help='그리드서치를 사용하여 하이퍼파라미터 최적화 수행')
    
    parser.add_argument('--scoring', type=str, default='roc_auc_ovr',
                        choices=['f1_weighted', 'roc_auc_ovr'],
                        help='그리드서치에 사용할 평가 지표')
    
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='그리드서치의 병렬 처리 작업 수 (-1은 모든 코어 사용)')
    
    # 앙상블 관련 파라미터
    parser.add_argument('--ensemble', action='store_true',
                    help='앙상블 수행')
    
    # 앙상블 유형
    parser.add_argument('--ensemble_type', type=str, default='voting', 
                        choices=['voting', 'stacking'],
                        help='앙상블 모델 유형 (voting, stacking)')
    
    # 보팅 앙상블 설정
    parser.add_argument('--voting_type', type=str, default='soft', 
                        choices=['hard', 'soft'],
                        help='보팅 방식 (hard 또는 soft)')
    
    # 앙상블에 포함할 모델 목록
    parser.add_argument('--ensemble_models', type=str, nargs='+', default=['xgb', 'gb'],
                        help='앙상블에 포함할 모델 목록 (공백으로 구분)')

    # 메타 모델 (스태킹 앙상블용)
    parser.add_argument('--meta_model', type=str, default='lr',
                        choices=['lr', 'lgbm', 'svm'],
                        help='스태킹 앙상블의 메타 모델')
    

    # 모델별 하이퍼파라미터
    # DecisionTree
    parser.add_argument('--dt_max_depth', type=int, default=None, 
                        help='DecisionTree의 최대 깊이')
    
    # RandomForest
    parser.add_argument('--rf_n_estimators', type=int, default=100, 
                        help='RandomForest의 트리 개수')
    parser.add_argument('--rf_max_depth', type=int, default=None, 
                        help='RandomForest의 최대 깊이')
    
    # GradientBoosting
    parser.add_argument('--gb_n_estimators', type=int, default=100, 
                        help='GradientBoosting의 트리 개수')
    parser.add_argument('--gb_learning_rate', type=float, default=0.1, 
                        help='GradientBoosting의 학습률')
    
    # XGBClassifier: grid search 후 roc_auc fix
    '''
    f1: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}
    roc_auc: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 0.8}
    '''
    parser.add_argument('--xgb_colsample_bytree', type=float, default=0.8)
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1)
    parser.add_argument('--xgb_max_depth', type=int, default=5)
    parser.add_argument('--xgb_min_child_weight', type=int, default=5)
    parser.add_argument('--xgb_n_estimators', type=int, default=200)
    parser.add_argument('--xgb_subsample', type=float, default=0.8)
    
    # SVM 파라미터
    parser.add_argument('--svm_C', type=float, default=1.0)
    parser.add_argument('--svm_kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'])
    parser.add_argument('--svm_gamma', type=str, default='scale', choices=['scale', 'auto'])

    # MLP 파라미터: grid search 후 roc_auc fix
    '''
    f1: {'alpha': 0.001, 'hidden_layer_sizes': (100,), 'max_iter': 500}
    roc_auc: {'alpha': 0.01, 'hidden_layer_sizes': (100, 50), 'max_iter': 200}
    '''    
    parser.add_argument('--mlp_alpha', type=float, default=0.01)
    parser.add_argument('--mlp_hidden_layers', type=tuple, default=(100, 50))
    parser.add_argument('--mlp_max_iter', type=int, default=200)
    
    return parser.parse_args()

def create_model_configs(args):
    """
    주어진 설정(args)을 기반으로 앙상블 모델별 개별 설정을 생성
    """
    model_configs = []
    
    base_args = deepcopy(args)
    
    for model_name in args.ensemble_models:
        model_args = deepcopy(base_args)
        model_args.model = model_name
        
        model_configs.append((model_name, model_args))
    
    return model_configs