# Smart Pandas

`SmartPandas` is a Pandas api extension for defining machine learning workflow metadata, via a config file, for Pandas dataframes. The goal of the package is to simplify the data flow for end to end ML pipelines by having clear definitions for columns, and allowing easy accessability of column groups at all times.



## Simple Example
Define a dataframe config in a `.yaml` file.
```yaml
{
  name: "life_expectancy_modelling_data",
  columns: [
    {
      name: "user_id",
      schema: {"dtype": "str"},
      labels: ["unique_identifier"],
      description: "Unique identifier for the person"
    },
    {
      name: "name",
      schema: {"dtype": "str"},
      labels: ["metadata"],
      description: "Name of the person"
    },
    {
      name: "weight",
      schema: {"dtype": "float"},
      labels: ["raw_feature"],
      description: "Weight of the person in kg"
    },
    {
      name: "height",
      schema: {"dtype": "float"},
      labels: ["raw_feature"],
      description: "Height of the person in cm"
    },
    {
      name: "age",
      schema: {"dtype": "int"},
      labels: ["raw_feature", "model_feature"],
      description: "Age of the person in years"
    },
    {
      name: "bmi",
      schema: {"dtype": "float"},
      labels: ["derived_feature", "model_feature"],
      description: "BMI of the person"
    },
    {
      name: "life_expectancy",
      schema: {"dtype": "int"},
      labels: ["target"],
      description: "Life expectancy of the person in years"
    },
  ]
}
```

Then import pandas from the smart pandas library, initialise the config, and you're away!

```python
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

print(data.config.raw_features)

#
#        weight  height  age
# 0      78     180   31
# 1      74     182   31
# 2      80     185   34
#
```