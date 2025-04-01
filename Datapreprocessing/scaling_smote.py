from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN

class DataScalerAndSampler:
    def __init__(self, dataframe, method='smote', random_state=0):
        self.dataframe = dataframe
        self.random_state = random_state
        self.method = method.lower()  # 'smote' 또는 'adasyn'
        self.scaler = MinMaxScaler()
        
        # 초기화시 샘플링 메서드는 나중에 설정
        self.sampler = None
        
        self.X_scaled = None
        self.y_scaled = None
        self.X_resampled = None
        self.y_resampled = None
        self.X_inverse = None

    def scale_data(self):
        X = self.dataframe.iloc[:, 1:-2]
        y_scaled = self.dataframe.iloc[:, -1:]
        self.X_scaled = self.scaler.fit_transform(X)
        self.y_scaled = y_scaled

    def apply_sampling(self):
      
        # 라벨 1과 3의 현재 샘플 수 계산
        label_counts = self.dataframe['Label'].value_counts()
        count_label_1 = label_counts.get(1, 0)
        count_label_3 = label_counts.get(3, 0)
        
        # 각 라벨에 10000개씩 더한 값으로 sampling_strategy 설정
        target_count_1 = count_label_1 + 10000
        target_count_3 = count_label_3 + 10000
        
        sampling_strategy = {1: target_count_1, 3: target_count_3}
        
        # 선택된 샘플링 방법에 따라 객체 생성
        if self.method == 'smote':
            self.sampler = SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
        elif self.method == 'adasyn':
            self.sampler = ADASYN(random_state=self.random_state, sampling_strategy=sampling_strategy)
        else:
            raise ValueError("Unknown sampling method. Use 'smote' or 'adasyn'.")
        
        # 오버샘플링 적용
        self.X_resampled, self.y_resampled = self.sampler.fit_resample(self.X_scaled, self.y_scaled)

    def inverse_scale(self, X_scaled):
        self.X_inverse = self.scaler.inverse_transform(X_scaled)
        return self.X_inverse

    def get_processed_data(self):
        return self.X_resampled, self.y_resampled, self.X_scaled, self.y_scaled