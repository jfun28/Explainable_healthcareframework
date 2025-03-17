class WeightChangeLabeler:
    def __init__(self, dataframe, threshold):
        self.dataframe = dataframe
        self.threshold = threshold

    def label_weight_changes(self):
        grouped = self.dataframe.groupby('ID')
        self.dataframe['Label'] = None

        for _, group in grouped:
            weight_changes = group['weight'].diff().shift(-1)
            labels = weight_changes.apply(lambda x: 3 if x > self.threshold else 1 if x < -self.threshold else 2)
            self.dataframe.loc[group.index, 'Label'] = labels

        self.dataframe['Label'].fillna(0, inplace=True)
        return self.dataframe
