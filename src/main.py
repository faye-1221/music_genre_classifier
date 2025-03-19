if __name__ == "__main__":
    from config import parse_arguments, create_model_configs
    from data_processor import DataProcessor
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from grid_search import GridSearchOptimizer
    from ensemble_model import EnsembleModelBuilder
    
    args = parse_arguments()
    
    # 데이터 로드 및 전처리
    df_x, df_y, label_encoder = DataProcessor.load_data()
    
    # 모델 생성
    model = ModelFactory.get_model(args)
    print(f"선택된 모델: {model.__class__.__name__}")
    
    if args.grid_search:
        print("그리드서치를 통한 하이퍼파라미터 최적화를 시작합니다...")
        param_grid = GridSearchOptimizer.get_param_grid_for_model(args.model)
        
        optimizer = GridSearchOptimizer(model, param_grid, label_encoder)
        best_model, best_params, results = optimizer.optimize(
            df_x, df_y, scoring=args.scoring, n_jobs=args.n_jobs
        )
        
        print(f"최적의 하이퍼파라미터: {best_params}")
        print(f"최적 모델을 사용하여 K-fold 교차 검증을 수행합니다...")
        
        # 최적화된 모델로 학습
        # trainer = ModelTrainer(best_model, label_encoder, k=args.k)
        # trainer.train_with_kfold(df_x, df_y)

    elif args.ensemble:
        model_configs = create_model_configs(args)

        if args.ensemble_type == 'voting':
            ensemble = EnsembleModelBuilder.create_voting_ensemble(
                model_configs, 
                voting=args.voting_type
            )
            print(f"보팅 방식: {args.voting_type}")
        
        elif args.ensemble_type == 'stacking':
            ensemble = EnsembleModelBuilder.create_stacking_ensemble(
                model_configs,
                meta_model=args.meta_model
            )
            print(f"메타 모델: {args.meta_model}")
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 유형: {args.ensemble_type}")
        
        trainer = ModelTrainer(ensemble, label_encoder, k=args.k)
        trainer.train_with_kfold(df_x, df_y)

    else:
        # 단일 모델 학습
        trainer = ModelTrainer(model, label_encoder, k=args.k)
        trainer.train_with_kfold(df_x, df_y)