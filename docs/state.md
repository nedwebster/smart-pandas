# State Management in Smart Pandas

## Overview

The state management system in Smart Pandas provides a sophisticated way to track and understand where your data is in the processing pipeline. The `State` class automatically infers the current state of your data by analyzing the relationship between the columns present in your DataFrame and the columns specified in your configuration.

This system is particularly valuable for:
- **Pipeline Validation**: Ensuring data is in the expected state before processing
- **Automatic Processing**: Determining what transformations are needed
- **Error Detection**: Identifying corrupted or incomplete datasets
- **Workflow Management**: Managing complex ML pipelines with multiple stages

## Core Concepts

### State Names

The `StateName` defines the different stages which data can exist in:

**RAW**: Data contains original features but lacks processed/engineered features

**PROCESSED**: Data contains model-ready features and all required columns

**UNKNOWN**: Data state cannot be determined from available information

**CORRUPTED**: Data is missing critical columns or has structural issues

### ML Stages

The `MLStage` defines the machine learning pipeline context:


**TRAINING**: Data includes target variables and is suitable for model training
**INFERENCE**: Data lacks target variables and is prepared for prediction


## Integration with Smart Pandas

The state system integrates seamlessly with other Smart Pandas components:

- **Configuration**: Uses `DataConfig` to understand expected column structure
- **Tags**: Leverages tag information to determine column roles
- **Validation**: Provides automatic validation of data integrity

This state management system provides a robust foundation for building intelligent, self-aware data processing pipelines that can adapt to different data conditions and pipeline stages.