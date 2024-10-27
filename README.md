# Smart Pandas

`SmartPandas` is framework for defining machine learning workflow metadata, via a config, file for Pandas dataframes. The goal of the package is to simplify data flow for end to end ML pipelines by having clear definitions for columns for columns.



## Simple Example
Define a dataframe config in a `.yaml` file.
```yaml
{
  name: "life_expectancy_modelling_data",
  columns: [
    {
      name: "user_id",
      type: "str",
      labels: ["unique_identifier"],
      description: "Unique identifier for the person"
    },
    {
      name: "name",
      type: "str",
      labels: ["metadata"],
      description: "Name of the person"
    },
    {
      name: "weight",
      type: "float",
      labels: ["raw_feature"],
      description: "Weight of the person in kg"
    },
    {
      name: "height",
      type: "float",
      labels: ["raw_feature"],
      description: "Height of the person in cm"
    },
    {
      name: "age",
      type: "int",
      labels: ["raw_feature", "model_feature"],
      description: "Age of the person in years"
    },
    {
      name: "bmi",
      type: "float",
      labels: ["derived_feature", "model_feature"],
      description: "BMI of the person"
    },
    {
      name: "life_expectancy",
      type: "int",
      labels: ["target"],
      description: "Life expectancy of the person in years"
    },
  ]
}
```

Ingest the config in python using `SmartPandas`

```python
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
```
