# Tags System in Smart Pandas

## Overview

The tags system in Smart Pandas provides a flexible way to categorize and label columns in your datasets. Tags help define the role and purpose of each column, enabling better data management, validation, and automated processing workflows.

Tags define 4 key properties for the column that they are assigned to:
- `data_attribute_name`: The name of the data attribute that the tag is associated with. This name is what is used to access columns with that tag, eg: `data.smart_pandas.raw_features`, where `raw_features` is the data attribute name.
- `config_limit`: The upper and lower limit for the count of this tag in the config. Eg, a tag limit of (1, 2) means that there must be at least 1 and at most 2 columns with this tag in the config.
- `required`: Whether a column with this tag is required for a valid dataset. If the tag has this value set to `True`, then the column must be present in the dataframe at all times. Note, this is whether the column is required to be in the dataframe, not whether the tag is required to be in the config. For example, the `target` tag is required in the config based on its `config_limit` value, but the column itself is not always required in the dataframe.
- `compatible_with`: A set of tag names that this tag is compatible with.


## Available Tags

Smart Pandas comes with pre-defined tags, the user should not need to create their own tags. The available tags are:

| Tag Name | Data Attribute Name | Config Limit | Required | Compatible With |
|----------|-------------------|--------------|----------|----------------|
| `target` | `target` | (1, 1) | No | - |
| `raw_feature` | `raw_features` | (1, None) | No | `model_feature`, `row_timestamp` |
| `derived_feature` | `derived_features` | (None, None) | No | `model_feature` |
| `metadata` | `metadata` | (None, None) | No | `unique_identifier`, `row_timestamp` |
| `unique_identifier` | `unique_identifier` | (1, 1) | Yes | `metadata` |
| `model_feature` | `model_features` | (1, None) | No | `raw_feature`, `derived_feature` |
| `row_timestamp` | `row_timestamp` | (1, 1) | Yes | `raw_feature`, `metadata` |
| `weight` | `weight` | (None, 1) | No | - |

### Tag Descriptions

- **`target`**: The target variable for machine learning models. Exactly one column must have this tag.
- **`raw_feature`**: Features that come directly from the raw data source. At least one column must have this tag, with no upper limit.
- **`derived_feature`**: Features that are computed or derived from raw features. No limits on the number of columns.
- **`metadata`**: Columns containing metadata about the data records. No limits on the number of columns.
- **`unique_identifier`**: A column that uniquely identifies each row. Exactly one column must have this tag and must always be present in the dataset.
- **`model_feature`**: Features that are ready to be used by machine learning models. At least one column must have this tag, with no upper limit.
- **`row_timestamp`**: A timestamp column indicating when each row was created or recorded. Exactly one column must have this tag and must always be present in the dataset.
- **`weight`**: Sample weights for machine learning models. At most one column can have this tag.

### Config Limit Notation

- `(1, 1)`: Exactly 1 column must have this tag
- `(1, None)`: At least 1 column must have this tag, no upper limit
- `(None, 1)`: At most 1 column can have this tag, no lower limit
- `(None, None)`: No limits on the number of columns with this tag
