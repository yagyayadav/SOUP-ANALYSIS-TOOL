# Project Description — mimic3-benchmarks

## Device Overview
This software is a clinical decision support system based on the mimic3-benchmarks repository by YerevaNN. It trains and evaluates machine learning models on the MIMIC-III clinical dataset to support clinical predictions such as in-hospital mortality, length of stay, and phenotype classification.

## Medical Function
The system processes patient clinical data and produces risk predictions that are intended to assist clinicians in making treatment decisions. Incorrect predictions could lead to wrong clinical decisions and direct patient harm.

## Programming Languages
Python

## IEC 62304 Overall Safety Classification
Class C — failure of this software could contribute to serious injury or death due to its direct influence on clinical decision-making.

## Patient Safety Context
- The software output (model predictions) is used to support clinical decisions
- Incorrect model output could lead to wrong diagnosis or treatment
- Any failure in data preprocessing, feature extraction, or model inference that produces wrong values reaches the clinician without a safety net
- The system runs on standard CPU hardware with no hardware-level safety checks

## Key Safety-Critical Functions
- Patient data ingestion and preprocessing
- Feature extraction (including statistical computations)
- Model training and inference
- Output generation for clinical use

## Known Dependencies
Declared in requirements.txt. A Software Bill of Materials (SBOM) in CycloneDX JSON format is provided separately as input to this tool.

## How each library is used in source code

NumPy: Used across preprocessing and feature_extractor.py 
       for numerical array operations on patient data before model input.

SciPy: scipy.stats.skew called directly in 
       mimic3models/feature_extractor.py for skewness 
       feature computation fed into model input array.

PyYAML: Used in preprocessing scripts for loading 
        dataset configuration files.

pandas: Used across benchmark scripts for loading 
        and reshaping patient records from CSV files.

TensorFlow: Backend for all model training and inference 
            in mimic3models/ directory.

Keras: model.fit() and model.predict() called in 
       mimic3models/keras_utils.py for training and inference.