# model_trainer.py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from logger import Logger
from visualizer import Visualizer

class ModelTrainer:
    def __init__(self, model, label_encoder, k=5):
        """
        모델 학습기 초기화
        """
        self.model = model
        self.label_encoder = label_encoder
        self.k = k
        self.model_name = model.__class__.__name__
        self.logger, self.log_file = Logger.setup_logger(self.model_name)
    
    def train_with_kfold(self, df_x, df_y):
        """
        StratifiedKFold로 모델을 학습
        """
        self.logger.info(f"Starting {self.k}-fold cross validation for {self.model_name}")
        self.logger.info(f"Data shape - X: {df_x.shape}, y: {len(df_y)}")

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1)

        val_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
        }
        test_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
        }

        # 클래스 분포
        # unique_classes, class_counts = np.unique(df_y, return_counts=True)
        # self.logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")

        for fold, (train_index, test_index) in enumerate(skf.split(df_x, df_y)):  # df_y도 전달
            fold_num = fold + 1
            self.logger.info(f"Starting fold {fold_num}/{self.k}")
            
            X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
            y_train, y_test = df_y[train_index], df_y[test_index]
            
            # 테스트 데이터의 클래스 분포
            # test_unique, test_counts = np.unique(y_test, return_counts=True)
            # self.logger.info(f"Test set class distribution: {dict(zip(test_unique, test_counts))}")

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, shuffle=True, random_state=1, stratify=y_train
            )
        
            self.logger.info(f"Fold {fold_num} - Training set: {X_train.shape[0]}, "
                             f"Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}")
            
            self.model.fit(X_train, y_train)
            self.logger.info(f"Fold {fold_num} - Model training completed")

            # Validation 메트릭 계산
            val_metrics_fold = self._calculate_metrics(X_val, y_val, "Validation", fold_num)
            for metric_name, value in val_metrics_fold.items():
                val_metrics[metric_name].append(value)
            
            # Test 메트릭 계산
            test_metrics_fold = self._calculate_metrics(X_test, y_test, "Test", fold_num)
            for metric_name, value in test_metrics_fold.items():
                test_metrics[metric_name].append(value)
            
            # 현재 폴드의 분류 리포트
            pred_test = self.model.predict(X_test)
            Visualizer.plot_confusion_matrix(
                y_test, pred_test, self.label_encoder, self.logger, self.log_file, fold_num
            )
        
        # 평균 메트릭 계산 및 로깅
        self._log_average_metrics(val_metrics, "Avg_Validation")
        self._log_average_metrics(test_metrics, "Avg_Test")
        
        self.logger.info("Cross-validation completed")
        
        # 학습 곡선 그리기
        # Visualizer.plot_learning_curve(self.model, df_x, df_y, self.logger, self.log_file)
    
    def _calculate_metrics(self, X, y, phase, fold_num=None):
        """
        주어진 데이터에 대한 모델 성능 메트릭을 계산
        """
        try:
            pred = self.model.predict(X)
            
            # 클래스 분포 확인
            # unique_y, counts_y = np.unique(y, return_counts=True)
            # unique_pred, counts_pred = np.unique(pred, return_counts=True)
            
            # self.logger.info(f"{phase} actual class distribution: {dict(zip(unique_y, counts_y))}")
            # self.logger.info(f"{phase} predicted class distribution: {dict(zip(unique_pred, counts_pred))}")
            
            metrics = {
                'accuracy': accuracy_score(y, pred),
                'precision': precision_score(y, pred, average='weighted', zero_division=0),
                'recall': recall_score(y, pred, average='weighted', zero_division=0),
                'f1': f1_score(y, pred, average='weighted')
            }
            
            # AUC 계산 (예측 확률이 있는 경우)
            auc = 0
            if hasattr(self.model, "predict_proba"):
                try:
                    y_proba = self.model.predict_proba(X)
                    # AUC 계산 시 클래스 개수 확인
                    n_classes = len(np.unique(y))
                    if y_proba.shape[1] != n_classes:
                        self.logger.warning(f"AUC calculation: predict_proba has {y_proba.shape[1]} columns, but there are {n_classes} classes")
                    
                    # 다중 클래스 AUC 계산
                    auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                    metrics['auc'] = auc
                except Exception as e:
                    self.logger.warning(f"AUC calculation failed for {phase}: {e}")
                    metrics['auc'] = 0
            
            # 메트릭 로깅
            Logger.log_metrics(self.logger, phase, fold_num, **metrics)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {phase}: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}
    
    def _log_average_metrics(self, metrics_dict, phase):
        """
        평균 메트릭을 계산하고 로깅
        """
        avg_metrics = {}
        for metric_name, values_list in metrics_dict.items():
            avg_metrics[metric_name] = np.mean(values_list)
        
        Logger.log_metrics(self.logger, phase, None, **avg_metrics)