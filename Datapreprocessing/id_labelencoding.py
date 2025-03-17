from sklearn.preprocessing import LabelEncoder
import pandas as pd

class LabelEncoderWrapper:
    def __init__(self, dataframe, column_name):
        self.dataframe = dataframe
        self.column_name = column_name
        self.encoder = LabelEncoder()

    def fit(self):
        self.encoder.fit(self.dataframe[self.column_name])

    def transform(self):
        self.dataframe[self.column_name] = self.encoder.transform(self.dataframe[self.column_name])

    def inverse_transform(self):
        self.dataframe[self.column_name] = self.encoder.inverse_transform(self.dataframe[self.column_name])



    