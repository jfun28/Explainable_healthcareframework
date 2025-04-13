import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        """
        데이터프레임을 전처리하는 클래스
        
        Parameters:
        df (pandas.DataFrame): 처리할 데이터프레임
        """
        self.df = df.copy()  # 원본 데이터 변경 방지
        self.preprocess_data()
        
    def preprocess_data(self):
        """전체 전처리 파이프라인을 실행"""
        self.clean_height_weight()  # 키와 체중 데이터 정리
        self.interpolate_data()     # 보간법으로 결측치 처리
        self.handle_outliers()      # 이상치 처리
        
    def clean_height_weight(self):
        """ID별로 키와 체중 데이터의 기본 클리닝 수행"""
        temp = []
        
        for uid in self.df['ID'].unique():
            # ID별로 데이터 추출
            user_df = self.df[self.df['ID'] == uid].reset_index(drop=True)
            
            # 기본적인 ffill/bfill 수행
            for col in ['height', 'weight']:
                if col in user_df.columns:
                    # NaN 값이 있는 경우에만 처리
                    if user_df[col].isna().any():
                        # 앞/뒤 값으로 채우기를 조합하여 사용
                        user_df[col] = user_df[col].ffill().bfill()
            
            temp.append(user_df)
            
        # 처리된 데이터프레임 결합
        self.df = pd.concat(temp, axis=0).reset_index(drop=True)
    
    def interpolate_data(self):
        """시계열 데이터에 보간법 적용"""
        temp = []
        
        for uid in self.df['ID'].unique():
            # ID별로 데이터 추출
            user_df = self.df[self.df['ID'] == uid].reset_index(drop=True)
            
            # 날짜나 시간 열이 있다면 그것을 기준으로 정렬
            # 예시: '날짜' 열이 있다면 정렬
            time_cols = [col for col in user_df.columns if any(time_word in col.lower() for time_word in ['date', 'time', 'day'])]
            
            if time_cols:
                user_df = user_df.sort_values(by=time_cols[0]).reset_index(drop=True)
            
            # 수치형 열에 대해 보간법 적용
            numeric_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                # 'ID' 열은 제외
                if col != 'ID' and user_df[col].isna().any():
                    # 선형 보간법 적용
                    user_df[col] = user_df[col].interpolate(method='linear')
            
            temp.append(user_df)
        
        # 처리된 데이터프레임 결합
        self.df = pd.concat(temp, axis=0).reset_index(drop=True)
    
    def calculate_growth_percentage(self, first, last):
        """두 값 사이의 성장률(%) 계산"""
        if first == 0:
            return 0  # 0으로 나누기 방지
        
        growth_percent = (last - first) / first * 100
        return abs(growth_percent)
    
    def handle_outliers(self):
        """키 데이터의 이상치 처리"""
        temp = []
        
        for uid in self.df['ID'].unique():
            # ID별로 데이터 추출
            user_df = self.df[self.df['ID'] == uid].reset_index(drop=True)
            
            # 키 이상치 처리
            if 'height' in user_df.columns and len(user_df) > 1:
                for j in range(1, len(user_df)):
                    # 성장은 감소하지 않는다는 가정 (키 데이터)
                    if user_df.at[j, 'height'] < user_df.at[j-1, 'height']:
                        user_df.at[j, 'height'] = user_df.at[j-1, 'height']
                    
                    # 비현실적인 급격한 성장 감지 (7% 초과)
                    growth = self.calculate_growth_percentage(user_df.at[j-1, 'height'], user_df.at[j, 'height'])
                    if growth > 7:
                        # 이전 값과 현재 값의 가중 평균으로 대체 (급격한 변화 완화)
                        user_df.at[j, 'height'] = (user_df.at[j-1, 'height'] * 0.7) + (user_df.at[j, 'height'] * 0.3)
            
            temp.append(user_df)
        
        # 처리된 데이터프레임 결합
        self.df = pd.concat(temp, axis=0).reset_index(drop=True)
    
    def get_processed_data(self):
        """처리된 최종 데이터프레임 반환"""
        return self.df




