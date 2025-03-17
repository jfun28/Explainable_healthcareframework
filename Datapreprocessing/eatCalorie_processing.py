import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class BigCalorieScaler:
    def __init__(self, upper_threshold=4000):
        self.upper_threshold = upper_threshold

    def minmax_scaler(self, data):
        """
        Perform Min-Max Scaling on the provided data.
        
        Args:
        data: A 1-dimensional list or numpy array.
        
        Returns:
        Numpy array after applying Min-Max scaling.
        """
        data = np.array(data)
        min_val = np.max([4000, np.min(data)])  # Ensuring min_val is at least 4000 or higher
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    def scale_calories(self, df):
        df_eat_calorie = df[df['eat calorie'].notna()].reset_index(drop=True)
        df_eat_calorie['MinMaxScaler'] = self.minmax_scaler(df_eat_calorie['eat calorie'])
        return pd.concat([df, df_eat_calorie[['MinMaxScaler']]], axis=1)


    def process_groups(self, df):
        """
        Process and scale calorie data for each group in the dataframe.

        Args:
        df: DataFrame with multiple groups.

        Returns:
        DataFrame after processing.
        """
        grouped = df.groupby('ID')
        df_total = pd.DataFrame()
        
        for _, group in grouped:
            df_group = self.scale_calories(group)
            
            for i in range(len(group)):
                if pd.isna(df_group["eat calorie"].iloc[i]):
                    continue
                
                if df_group["eat calorie"].iloc[i] > self.upper_threshold:
                    scale_factor = 1 + df_group["MinMaxScaler"].iloc[i]
                    df_group["eat calorie"].iloc[i] = self.upper_threshold * scale_factor

            df_total = pd.concat([df_total, df_group], ignore_index=True)
        
        return df_total

# Example usage:
# scaler = CalorieScaler()
# df_processed = scaler.process_groups(df)
