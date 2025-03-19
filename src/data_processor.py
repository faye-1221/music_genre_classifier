import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer

class DataProcessor:
    @staticmethod
    def load_data(file_path="../data/genres_v2.csv"):
        """
        데이터를 로드하고 전처리
        """
        df = pd.read_csv(file_path, low_memory=False)

        # 불필요한 feature 제거
        df = df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title', 'key', 'mode', 'time_signature'], axis=1)

        df_y = df['genre']
        df_x = df.drop(['genre'], axis=1)

        # Underground Rap, Pop 클래스 제거
        mask = ~df_y.isin(['Underground Rap', 'Pop'])
        df_y = df_y[mask]
        df_x = df_x.loc[mask]
        df_y = df_y.reset_index(drop=True)
        df_x = df_x.reset_index(drop=True)

        # Label Encoding
        label_encoder = LabelEncoder()
        df_y_encoded = label_encoder.fit_transform(df_y)

        # 이상치 처리
        df_x = DataProcessor.impute_outliers(df_x, df_y_encoded)

        # 스케일링 적용
        quantile_scaler = QuantileTransformer(output_distribution='normal')
        df_x = pd.DataFrame(quantile_scaler.fit_transform(df_x), columns=df_x.columns)

        return df_x, df_y_encoded, label_encoder
    
    @staticmethod
    def outlier_mean(df_x, df_y, class_label):
        """
        특정 클래스의 피처 평균값을 계산
        """
        return df_x[df_y == class_label].mean()

    @staticmethod
    def impute_outliers(df_x, df_y):
        """
        이상치를 탐지하고 같은 클래스의 평균값으로 대체
        """
        threshold = 3
        z_scores = np.abs(zscore(df_x))
        outliers_indices = np.where(z_scores > threshold)[0]
        df_x_imputed = df_x.copy()
        
        for idx in np.unique(outliers_indices):
            class_label = df_y[idx]
            imputed_values = DataProcessor.outlier_mean(df_x, df_y, class_label)
            for col in df_x.columns:
                df_x_imputed.at[idx, col] = np.array(imputed_values[col]).astype(df_x[col].dtype)
        
        return df_x_imputed
