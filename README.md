<img src="docs/images/logo.png" alt="Smart Pandas Logo" width="200"/>

# Smart Pandas


`smart-pandas` provides a Pandas api extension for integrating machine learning data schemas via a config file and the open source Pandera package. The goal of the package is to simplify the data flow for end to end ML pipelines by having clear definitions for columns, and allowing easy accessability of column groups at all times.

Data validation is an important part of any data intesive application, and that includes machine learning projects. However most schema frameworks lack the flexibiltiy and complexity to appropriately define schemas for ML datasets. As data flows through your ML pipeline, you code may add and/or remove columns at various stages, making a static schema difficult to maintain. Often you'll end up with various sets of column groupings with some semantic meaning, but no clear idea of what is available in the data at any given point in time. Smart Pandas solves this problem by allowing you to define a schema upfront which groups your columns into meaningful ML categories, and which can be dynamic based on the current stage of the ML process. Below are the two key things that `smart-pandas` provides:

1. Semantic groupings of your data columns through `tags`, eg: `raw_features`, `model_features`, etc., and a convenient way to access those columns in your dataframe through the pandas api extension.
2. A tracked `state` of your dataframe based on the available columns, and the ability to dynamically build your Pandera schemas based on that state. For example, your schema may differ in your training and inference pipelines.


![Data Flow](docs/images/data_flow.png)

The diagram above shows a simple example of the different semantic 'groups' of columns you may want moving through your ML data pipeline. Those groups can vary from stage to stage, and can even be different depending on which ML lifecycle stage you're at.


## Basic Usage

To use `smart-pandas`, all you need to do is swap your Pandas import to come form `smart-pandas`, which will give you access to the pandas api extension.

```python
from smart_pandas imnport pandas as pd
```
Next, you need to define your config as a `.yaml` file. The config contains definitions for each column in your datast. See the config documentation for more details on building your own config.

```yaml
{
  name: "life_expectancy_modelling_data",
  columns: [
    {
      name: "user_id",
      data_schema: {"dtype": "str"},
      tags: ["unique_identifier"],
      description: "Unique identifier for the person"
    },
    {
      name: "timestamp",
      data_schema: {"dtype": "datetime"},
      tags: ["row_timestamp"],
      description: "Timestamp of the data"
    },
    {
      name: "name",
      data_schema: {"dtype": "str"},
      tags: ["metadata"],
      description: "Name of the person"
    },
    {
      name: "weight",
      data_schema: {"dtype": "float"},
      tags: ["raw_feature"],
      description: "Weight of the person in kg"
    },
    {
      name: "height",
      data_schema: {"dtype": "float"},
      tags: ["raw_feature"],
      description: "Height of the person in cm"
    },
    {
      name: "age",
      data_schema: {"dtype": "int"},
      tags: ["raw_feature", "model_feature"],
      description: "Age of the person in years"
    },
    {
      name: "bmi",
      data_schema: {"dtype": "float"},
      tags: ["derived_feature", "model_feature"],
      description: "BMI of the person"
    },
    {
      name: "life_expectancy",
      data_schema: {"dtype": "int"},
      tags: ["target"],
      description: "Life expectancy of the person in years"
    },
  ]
}
```

Then you can initialise your `smart-pandas` config with your pandas dataframe, and begin accessing the `smart-pandas` attributes.

```python
data = pd.DataFrame(
    {
        "user_id": ["1", "2", "3"],
        "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
        "name": ["Emily", "Adam", "Charles"],
        "weight": [60, 74, 80],
        "height": [165, 182, 185],
        "age": [25, 30, 35],
        "life_expectancy": [90, 80, 80],
    }
)

data.smart_pandas.load_config(config_path="examples/example_config.yaml")

print(data.smart_pandas.raw_features)

#
#        weight  height  age
# 0      78     180   31
# 1      74     182   31
# 2      80     185   34
#
```

## Data Attributes
Data attributes are the semantic groupings of columns in your dataframe, and are defined by the `tags` in your config file. For more information on the available tags, see the [tags documentation](docs/tags.md). Currently available data attributes are:
- `raw_features`
- `derived_features`
- `model_features`
- `target`
- `unique_identifier`
- `row_timestamp`
- `metadata`
- `weights`

## State
`smart-pandas` tracks the synchronisation between the data columns and the configuration file through the `state` attribute. The `state` attribute represents a high level view of the data at certain phases in the ML data lifecycle. The state is built up of two attributes, the `StateName` and the `MLStage`. The `StateName` represents the current point in a specific data pipeline, whereas the `MLStage` identifies which data pipeline we are in. See the [state documentation](docs/state.md) for more details.

```python
data.smart_pandas.state

# StateName.RAW, MLStage.TRAINING
```

## Data Validation
`smart-pandas` uses Pandera for building data schemas to validate the data against. `smart-pandas` dynamically builds the schema based on the current state of the data, and the definitions in the config file.

```python
data.smart_pandas.validate(inplace=True)
```

## Development
The package uses uv for it's package management, and pytest for it's unit test framework. To run unit-tests, use the following code:

```shell
uv run pytest tests/.
```
