from smart_pandas import pandas as pd


data = pd.DataFrame(
    {
        "user_id": ["1", "2", "3"],
        "name": ["Ned", "Roland", "Tom"],
        "weight": [78, 74, 80],
        "height": [180, 182, 185],
        "age": [31, 31, 34],
        "life_expectancy": [80, 80, 80],
    }
)
data.smart_pandas.init(config_path="examples/example_config.yaml")

print(data.smart_pandas.raw_features)
"""
       weight  height  age
0      78     180   31
1      74     182   31
2      80     185   34
"""
