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
      tags: ["unique_identifier"],
      description: "Unique identifier for the person"
    },
    {
      name: "name",
      schema: {"dtype": "str"},
      tags: ["metadata"],
      description: "Name of the person"
    },
    {
      name: "weight",
      schema: {"dtype": "float"},
      tags: ["raw_feature"],
      description: "Weight of the person in kg"
    },
    {
      name: "height",
      schema: {"dtype": "float"},
      tags: ["raw_feature"],
      description: "Height of the person in cm"
    },
    {
      name: "age",
      schema: {"dtype": "int"},
      tags: ["raw_feature", "model_feature"],
      description: "Age of the person in years"
    },
    {
      name: "bmi",
      schema: {"dtype": "float"},
      tags: ["derived_feature", "model_feature"],
      description: "BMI of the person"
    },
    {
      name: "life_expectancy",
      schema: {"dtype": "int"},
      tags: ["target"],
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
data.smart_pandas.init(config_path="examples/example_config.yaml")

print(data.smart_pandas.raw_features)

#
#        weight  height  age
# 0      78     180   31
# 1      74     182   31
# 2      80     185   34
#
```