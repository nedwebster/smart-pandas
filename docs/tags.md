# Tags System in Smart Pandas

## Overview

The tags system in Smart Pandas provides a flexible way to categorize and label columns in your datasets. Tags help define the role and purpose of each column, enabling better data management, validation, and automated processing workflows.

## How Tags Work

Each tag in Smart Pandas is represented by a class that inherits from the base `Tag` class. Tags have three main properties:

- **name**: A unique string identifier for the tag
- **compatible_with**: A list of other tags that can coexist with this tag on the same column
- **dataset_limit**: An optional limit on how many columns in a dataset can have this tag

### Tag Compatibility

Tags can be compatible with other tags, meaning a single column can have multiple tags assigned to it. For example, a column tagged as `raw_feature` can also be tagged as `model_feature` since these tags are compatible.

### Dataset Limits

Some tags have dataset limits, meaning only a certain number of columns in a dataset can have that tag. For example, the `target` tag has a limit of 1, ensuring that only one column per dataset can be designated as the target variable.

## Available Tags

### target

**Purpose**: Identifies the target variable (dependent variable) in your dataset - the variable you want to predict or analyze.

**Use cases**:
- Binary classification target (0/1, True/False)
- Multi-class classification target
- Regression target (continuous values)
- Survival analysis endpoints

**Dataset limit**: 1 (only one target column per dataset)

### raw_feature

**Purpose**: Marks columns that contain original, unprocessed data directly from the source.

**Use cases**:
- Original sensor readings
- Raw survey responses
- Unprocessed text data
- Original numerical measurements

**Compatible with**: `model_feature`, `row_timestamp`

### derived_feature

**Purpose**: Identifies columns that have been created through feature engineering or data transformation from other columns.

**Use cases**:
- Calculated ratios or percentages
- Aggregated statistics
- Encoded categorical variables
- Polynomial features
- Principal components

**Compatible with**: `model_feature`

### model_feature

**Purpose**: Designates columns that will be used as input features for machine learning models.

**Use cases**:
- Features selected for training
- Preprocessed input variables
- Engineered features ready for modeling

**Compatible with**: `raw_feature`, `derived_feature`

### metadata

**Purpose**: Marks columns that contain information about the data itself rather than features for analysis.

**Use cases**:
- Data collection timestamps
- Source system identifiers
- Data quality flags
- Processing batch IDs

**Compatible with**: `unique_identifier`, `row_timestamp`

### unique_identifier

**Purpose**: Identifies the column that uniquely identifies each row in the dataset.

**Use cases**:
- Primary key columns
- Customer IDs
- Transaction IDs
- Record identifiers

**Compatible with**: `metadata`
**Dataset limit**: 1 (only one unique identifier per dataset)

### row_timestamp

**Purpose**: Marks the column that contains the timestamp when each row's data was created or recorded.

**Use cases**:
- Transaction timestamps
- Event occurrence times
- Data collection dates
- Record creation times

**Compatible with**: `raw_feature`, `metadata`
**Dataset limit**: 1 (only one row timestamp per dataset)

### weight

**Purpose**: Identifies a column that contains weights for observations, used in weighted analysis or modeling.

**Use cases**:
- Sampling weights
- Importance weights
- Frequency weights
- Inverse probability weights

**Dataset limit**: 1 (only one weight column per dataset)
