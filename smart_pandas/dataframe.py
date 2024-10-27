from smart_pandas.column_set import ColumnSet


class SmartDataFrame:
    def __init__(self, name: str, column_set: ColumnSet):
        self.name = name
        self.column_set = column_set

    @property
    def raw_features(self):
        return [
            column.name for column in self.column_set if "raw_feature" in column.labels
        ]

    @property
    def derived_features(self):
        return [
            column.name
            for column in self.column_set
            if "derived_feature" in column.labels
        ]

    @property
    def model_features(self):
        return [
            column.name
            for column in self.column_set
            if "model_feature" in column.labels
        ]

    @property
    def target(self):
        return [
            column.name for column in self.column_set if "target" in column.labels
        ] or None
