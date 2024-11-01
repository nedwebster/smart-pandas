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
data.config.init(config_path="examples/example_config.yaml")
data["bmi"] = data["weight"] / (data["height"] / 100) ** 2

print(type(data.config.schema))  # smart-pandas constructs a pandera DataFrameSchema
"""
<class 'pandera.api.pandas.container.DataFrameSchema'>
"""

validated_data = (
    data.config.validate()
)  # runs validation using the pandera DataFrameSchema

# validated_data still has the smart-pandas config attached
print(validated_data.config.raw_features)
"""
       weight  height  age
0      78     180   31
1      74     182   31
2      80     185   34
"""
