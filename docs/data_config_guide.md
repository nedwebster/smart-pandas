# DataConfig User Guide

## Overview

The DataConfig is the cornerstone of smart-pandas, providing a structured way to define and validate your dataset schemas. It allows you to specify column metadata, data types, and semantic tags that enable intelligent data processing and feature extraction.

## Why Use DataConfig?

- **Schema Validation**: Ensure your data matches expected types and constraints
- **Semantic Tagging**: Classify columns by their role (features, targets, metadata, etc.)
- **Automated Feature Extraction**: Automatically extract different types of features based on tags
- **Documentation**: Self-documenting datasets with descriptions and metadata
- **Reproducibility**: Consistent data processing across different environments

## Configuration File Structure

DataConfig uses YAML files to define dataset schemas. Here's the basic structure:

```yaml
name: your_dataset_name
columns:
  - name: column_name
    description: Human-readable description
    data_schema:
      dtype: data_type
      ...
    tags:
      - tag_name
      - ...
```

## Column Definition

Each column in your dataset should be defined with the following properties:

### Required Properties

- **name**: The exact column name as it appears in your data
- **description**: A clear description of what the column represents
- **data_schema**: Type information for the column
- **tags**: List of semantic tags that classify the column's role

### Data Schema

The `data_schema` section can accept any valid pandera column schema values. For example:

```yaml
data_schema:
  dtype: int        # Data type: int, float, str, bool, etc.
  nullable: true    # Whether the column can contain null values (optional)
```

## Column Tags

Tags are the key to smart-pandas' intelligent processing. They classify columns by their semantic meaning:

### Core Tags

- **`raw_feature`**: Original features from your data source
- **`derived_feature`**: Features created through transformations or calculations
- **`model_feature`**: Features that should be used for modeling
- **`target`**: The variable you want to predict
- **`unique_identifier`**: Unique row identifiers (like primary keys)
- **`metadata`**: Descriptive information that doesn't affect modeling
- **`row_timestamp`**: Timestamp information for time-series data

### Tag Combinations

Columns can have multiple tags. For example, a column can be both a `raw_feature` and a `model_feature`:

```yaml
- name: age
  description: The age of the passenger
  data_schema:
    dtype: float
    nullable: true
  tags:
    - raw_feature
    - model_feature
```

## Complete Example: Titanic Dataset

Here's how the famous Titanic dataset is configured:

```yaml
name: titanic_modelling_data
columns:
  # Unique identifier
  - name: id
    description: A unique identifier for each passenger
    data_schema:
      dtype: int
    tags:
      - unique_identifier

  # Target variable
  - name: survived
    description: Whether the passenger survived or not
    data_schema:
      dtype: int
    tags:
      - target

  # Raw features that are also model features
  - name: pclass
    description: The class of the passenger
    data_schema:
      dtype: int
    tags:
      - raw_feature
      - model_feature

  - name: sex
    description: The sex of the passenger
    data_schema:
      dtype: str
    tags:
      - raw_feature
      - model_feature

  - name: age
    description: The age of the passenger
    data_schema:
      dtype: float
      nullable: true
    tags:
      - raw_feature
      - model_feature

  # Metadata columns
  - name: name
    description: The full name of the passenger
    data_schema:
      dtype: str
    tags:
      - metadata

  - name: ticket
    description: The ticket number for that passenger
    data_schema:
      dtype: str
    tags:
      - metadata

  # Derived feature
  - name: number_of_cabins
    description: Number of cabins the passenger and family had
    data_schema:
      dtype: int
    tags:
      - derived_feature
      - model_feature
```

## Using Your DataConfig

Once you've created your configuration file, you can load it into smart-pandas:

```python
from smart_pandas import SmartDataFrame

# Load your data and config
df = SmartDataFrame.from_config("path/to/your_config.yaml")

# Access different types of features
raw_features = df.config.raw_features
model_features = df.config.model_features
target = df.config.target
```