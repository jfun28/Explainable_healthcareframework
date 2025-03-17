import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SMOTEDataOrganizer:
    def __init__(self, original_data, scaled_data, oversampled_data, oversampled_labels):
        self.original_data = original_data
        self.scaled_data = scaled_data
        self.oversampled_data = oversampled_data
        self.oversampled_labels = oversampled_labels
        self.scaler = MinMaxScaler()
        self.scaled_data_inverse = None
        self.oversampled_data_inverse = None
        self.final_df = None

    def inverse_scale(self):
        self.scaled_data_inverse = self.scaler.inverse_transform(self.scaled_data)
        self.oversampled_data_inverse = self.scaler.inverse_transform(self.oversampled_data)

    def check_smote_effect(self):
        scaled_df = pd.DataFrame(self.scaled_data_inverse, columns=self.original_data.columns)
        oversampled_df = pd.DataFrame(self.oversampled_data_inverse, columns=self.original_data.columns)
        oversampled_df['Smote_check'] = oversampled_df.ne(scaled_df).any(axis=1).astype(int)
        oversampled_df["Label"] = self.oversampled_labels
        return oversampled_df

    def organize_data(self):
        df_smote_check = self.check_smote_effect()
        df_not_smote = df_smote_check[df_smote_check['Smote_check'] == 0].reset_index(drop=True)
        df_smote = df_smote_check[df_smote_check['Smote_check'] == 1].reset_index(drop=True)
        df_not_smote['ID'] = self.original_data['ID']
        df_smote['ID'] = '999'
        self.final_df = pd.concat([df_not_smote, df_smote], axis=0).reset_index(drop=True)
        return self.final_df
