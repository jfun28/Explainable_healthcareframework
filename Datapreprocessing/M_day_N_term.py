import pandas as pd

class RollingAveragesCalculator:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rolling_averages = None

    def calculate_averages(self, columns, m, n):
        results = []
        grouped = self.dataframe.groupby('ID')

        for _, group in grouped:
            for i in range(0, len(group), n):
                if i + m <= len(group):
                    averages = {col: group[col][i:i+m].mean() for col in columns}
                    averages['ID'] = group['ID'].iloc[0]
                    averages['Start_Day'] = i
                    results.append(averages)

        self.rolling_averages = pd.DataFrame(results)
        return self.rolling_averages

    def get_rolling_averages(self):
        return self.rolling_averages
