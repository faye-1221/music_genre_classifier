from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class ModelFactory:
    @staticmethod
    def get_model(args):
        """
        주어진 인자에 따라 모델을 생성
        """
        if args.model == 'dt':
            return DecisionTreeClassifier(max_depth=args.dt_max_depth, random_state=1)
        
        elif args.model == 'rf':
            return RandomForestClassifier(n_estimators=args.rf_n_estimators, 
                                        max_depth=args.rf_max_depth, 
                                        random_state=1,
                                        class_weight='balanced')
        
        elif args.model == 'gb':
            return GradientBoostingClassifier(n_estimators=args.gb_n_estimators, 
                                            learning_rate=args.gb_learning_rate, 
                                            random_state=1)
        
        elif args.model == 'xgb':
            # https://ysyblog.tistory.com/80
            return XGBClassifier(
                                colsample_bytree=args.xgb_colsample_bytree,
                                learning_rate=args.xgb_learning_rate,
                                max_depth=args.xgb_max_depth,
                                min_child_weight=args.xgb_min_child_weight,
                                n_estimators=args.xgb_n_estimators,
                                subsample=args.xgb_subsample,
                                eval_metric='mlogloss',
                                random_state=1)

        elif args.model == 'svm':
            # https://inuplace.tistory.com/600
            return SVC(C=args.svm_C, 
                    kernel=args.svm_kernel, 
                    gamma=args.svm_gamma,
                    probability=True,
                    random_state=1)
        
        elif args.model == 'mlp':
            # https://sjh9708.tistory.com/225
            return MLPClassifier(alpha=args.mlp_alpha,
                                hidden_layer_sizes=args.mlp_hidden_layers,
                                max_iter=args.mlp_max_iter,
                                random_state=1)

        else:
            raise ValueError(f"지원하지 않는 모델: {args.model}")