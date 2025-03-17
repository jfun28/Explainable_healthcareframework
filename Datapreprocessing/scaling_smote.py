from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

class DataScalerAndSmoter:
    def __init__(self, dataframe, random_state=0):
        self.dataframe = dataframe
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        # self.smote = SMOTE(random_state=self.random_state,sampling_strategy={1: 10464, 3: 11114})
        self.smote = SMOTE(random_state=self.random_state,sampling_strategy={1: 10000, 3: 10000})
        # self.smote = SMOTE()

        self.X_scaled = None
        self.y_scaled = None
        self.X_resampled = None
        self.y_resampled = None
        self.X_inverse = None

    def scale_data(self):
        X = self.dataframe.iloc[:, 1:-2]
        y_scaled=self.dataframe.iloc[:, -1:]
        self.X_scaled = self.scaler.fit_transform(X)
        self.y_scaled=y_scaled


    def apply_smote(self):
        self.X_resampled, self.y_resampled = self.smote.fit_resample(self.X_scaled, self.y_scaled)

    def get_processed_data(self):
        return self.X_resampled, self.y_resampled, self.X_scaled, self.y_scaled

   
    def inverse_scale(self,X_scaled):
        self.X_inverse = self.scaler.inverse_transform(X_scaled)
        return self.X_inverse

    def get_processed_data(self):
        return self.X_resampled, self.y_resampled, self.X_scaled, self.y_scaled
    
# day 순서 포함
class DataScalerAndSmoterDay:
    def __init__(self, dataframe, random_state=42):
        self.dataframe = dataframe
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        # self.smote = SMOTE(random_state=self.random_state,sampling_strategy={1: 10000, 3: 10000})
        self.smote = SMOTE(random_state=self.random_state)

        self.X_scaled = None
        self.y_scaled = None
        self.X_resampled = None
        self.y_resampled = None
        self.X_inverse = None

    def scale_data(self):
        X = self.dataframe.iloc[:, 1:-1]
        y_scaled=self.dataframe.iloc[:, -1:]
        self.X_scaled = self.scaler.fit_transform(X)
        self.y_scaled=y_scaled


    def apply_smote(self):
        self.X_resampled, self.y_resampled = self.smote.fit_resample(self.X_scaled, self.y_scaled)

    def get_processed_data(self):
        return self.X_resampled, self.y_resampled, self.X_scaled, self.y_scaled

   
    def inverse_scale(self,X_scaled):
        self.X_inverse = self.scaler.inverse_transform(X_scaled)
        return self.X_inverse

    def get_processed_data(self):
        return self.X_resampled, self.y_resampled, self.X_scaled, self.y_scaled
    