# Machine Learning Pipeline
## *title*
Name : Your name 
Date​: DD/MM/YY
## Data Overview
### Independent Features
* `describe the features`​: Describe the features
### Target Feature
* `Target Feature`: Describe the target score
### Unique Identifying Features
* `index/ID`: Unique ID or index
### Catergorical Features
* `Categorical features?`: types of categorical features
### Binary Features
* `Binary features?`: Description of binary features
### Numerical Features
* `Numerical Features`: Description of numerical features

# Sypnopsis of the problem. 
* **Regression/Classification**: For you to decide whether you want to use regression or classification

## Overview of Submitted folder
.
├── eda.ipnyb
├── __init__.py
├── data
│   └── score.db # removed
├── requirements.txt
├── results
│   └── MAE.csv
├── run.sh
└── src
    ├── __init__.py
    ├── ml_module
    │   ├── __init__.py
    │   ├── data_prep.py
    │   ├── eda_preprocessing.py
    │   ├── model_classification.py
    │   └── model_regression.py
    └── run.py

## Executing the pipeline
**Step 1) Data-preprocessing(eda_preprocessing.py)**
Imports the data from .db file, data is processed through the findings from EDA.ipnyb

**Step 2) Data Preparation (data-prep.py)**
Data is prepared by one-hot encoding categorical features. Target encoding was done for ordinal features. Numerical features will be pre-processed as required. The overall dataset is also train-test split into 80/20 split.

**Step 3 & 4) Hyper Parameter tuning for Regression & Classification**
All hyper parameters are tuned through gridsearchCV through a predefined range^1^.
The evaluation criteria for regression and classification gridsearch are based on MAE of final test (numerical and categorical (1-10)).

 ^1^ In gridsearch CV the predefined hyperparameters has gone through multiple iterations previously to derive the optimal range. The grid can be expanded up to users discretion

**Step 4 & 5) Results**
Taking the best hyperparams previously found in step 3 and 4. the best parameters are fed into the final model and the results saved as a txt file
-> Since Classification is evaluated on an ordinal MAE, to calculate the true MAE, take ordinal MAE * 10.

## Running of machine learning pipeline.
Machine learning model created with python 3 and bash script.
### Installing dependencies
Paste the following command on your bash terminal to download dependencies
```sh
pip install -r requirements.txt
```
### Running the Machine Learning Pipeline
Past the followin command on your bash terminal to grant permission to execute the 'run.sh' file
```sh
chmod +x run.sh
```
Paste the following command on the bash terminal to run the machine learning programme
```sh
./run.sh
```

This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## License
MIT
