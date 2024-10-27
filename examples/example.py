from smart_pandas.config_reader import ConfigReader


smart_dataframe = ConfigReader().read_config(path="examples/example_config.yaml")

print(smart_dataframe.name)
# life_expectancy_modelling_data

print(smart_dataframe.raw_features)
# ['weight', 'height', 'age']

print(smart_dataframe.derived_features)
# ['bmi']

print(smart_dataframe.model_features)
# ['age', 'bmi']

print(smart_dataframe.target)
# ['life_expectancy']
