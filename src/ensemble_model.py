from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from model_factory import ModelFactory
from lightgbm import LGBMClassifier

class EnsembleModelBuilder:
    @staticmethod
    def create_voting_ensemble(models_config, voting='soft'):
        """
        보팅 앙상블 모델 생성
        """
        estimators = []
        
        for i, (model_name, args) in enumerate(models_config):
            model = ModelFactory.get_model(args)
            estimators.append((f"{model_name}_{i}", model))
        
        return VotingClassifier(estimators=estimators, voting=voting)
    
    @staticmethod
    def create_stacking_ensemble(models_config, meta_model):
        """
        스태킹 앙상블 모델 생성
        """
        estimators = []
        
        for i, (model_name, args) in enumerate(models_config):
            model = ModelFactory.get_model(args)
            estimators.append((f"{model_name}_{i}", model))
        
        final_estimator = EnsembleModelBuilder._get_meta_model(meta_model)
        
        return StackingClassifier(
            estimators=estimators, 
            final_estimator=final_estimator,
            cv=StratifiedKFold(n_splits=5),
            stack_method='predict_proba'
        )
    
    @staticmethod
    def _get_meta_model(meta_model_type):
        """
        메타 모델 선택
        """
        if meta_model_type == 'lr':
            return LogisticRegression(max_iter=1000, random_state=1)
        elif meta_model_type == 'lgbm':
            return LGBMClassifier(random_state=1)
        elif meta_model_type == 'svm':
            return SVC(probability=True, random_state=1)
        else:
            # 기본
            return LogisticRegression(max_iter=1000, random_state=1)