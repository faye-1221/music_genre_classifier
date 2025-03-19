from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from logger import Logger

class GridSearchOptimizer:
    def __init__(self, model, param_grid, label_encoder, cv=5):
        """
        그리드서치를 통해 하이퍼파라미터를 최적화
        """
        self.model = model
        self.param_grid = param_grid
        self.label_encoder = label_encoder
        self.cv = cv
        self.model_name = model.__class__.__name__
        self.logger, self.log_file = Logger.setup_logger(f"{self.model_name}_gridsearch")
        
        # 스코어러 정의
        self.scorers = {
            'roc_auc_ovr': make_scorer(roc_auc_score, multi_class='ovr'),
            'f1_weighted': make_scorer(f1_score, average='weighted')
        }
        
    def optimize(self, X, y, scoring='roc_auc_ovr', n_jobs=-1):
        """
        최적의 하이퍼파라미터 찾기
        """
        self.logger.info(f"Starting GridSearch for {self.model_name}")
        self.logger.info(f"Parameter grid: {self.param_grid}")
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=self.cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # 결과를 데이터프레임으로 변환하여 저장
        results = pd.DataFrame(grid_search.cv_results_)
        results_file = self.log_file.replace('.log', '_results.csv')
        results.to_csv(results_file)
        self.logger.info(f"GridSearch results saved to {results_file}")
        
        # 파라미터 중요도 시각화
        self._visualize_param_importance(grid_search)
        
        return grid_search.best_estimator_, grid_search.best_params_, results
        
    def _visualize_param_importance(self, grid_search):
        """
        그리드서치 결과를 시각화
        """
        results = pd.DataFrame(grid_search.cv_results_)
        
        # 성능이 좋은 상위 결과만 선택
        top_results = results.sort_values('mean_test_score', ascending=False).head(10)
        
        # 파라미터별로 플롯 생성
        for param in self.param_grid.keys():
            param_name = f'param_{param}'
            if param_name in results.columns:
                plt.figure(figsize=(10, 6))
                # 파라미터 값과 성능 점수 간의 관계 시각화
                param_values = results[param_name].astype(str)
                grouped = results.groupby(param_name)['mean_test_score'].mean()
                
                plt.bar(grouped.index.astype(str), grouped.values)
                plt.xlabel(param)
                plt.ylabel('Mean Test Score')
                plt.title(f'{param} vs Performance')
                plt.xticks(rotation=45)
                
                # 저장
                image_path = self.log_file.replace('.log', f'_param_{param}.png')
                plt.savefig(image_path)
                plt.close()
                self.logger.info(f"Parameter importance plot saved to {image_path}")
        
        # 모든 파라미터 조합에 대한 결과 시각화
        plt.figure(figsize=(12, 8))
        plt.scatter(
            range(len(results)), 
            results['mean_test_score'], 
            alpha=0.7
        )
        plt.xlabel('Parameter Combination')
        plt.ylabel('Mean Test Score')
        plt.title('Performance of All Parameter Combinations')
        
        image_path = self.log_file.replace('.log', '_all_combinations.png')
        plt.savefig(image_path)
        plt.close()
        self.logger.info(f"All combinations plot saved to {image_path}")

    @staticmethod
    def get_param_grid_for_model(model_name):
        """
        모델별로 최적화할 파라미터 그리드를 반환
        """
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'xgb': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [200, 500, 1000]
            }
        }
        
        return param_grids.get(model_name, {})