# Machine Learning Pipeline (Predicting O'levels Grades )
## AIAP Batch 9 technical assessment submission
Name : Tan Kah Ming
NRIC : S9240905H
Date​: 2021-08-02
## Data Overview
### Independent Features
* `student-ID`​: Unique ID for each student
### Target Feature
* `final_test`: Student's O-levels mathematics examination score
### Unique Identifying Features
* `student_ID`: Unique ID for each student
### Catergorical Features
* `bag_color`: Color of student bag
* `CCA`: Enrolled CCA
* `Learning Style` : Primary Learning Style
* `Mode_of_transport`: Mode of transport to school
### Binary Features
* `Tuition`: Indication of whether the student has tuition
* `Direct Admission`: Mode of entering the school
* `gender`: Y or N
### Numerical Features
* `attendance rate`: Attendance rate of the student (%)
* `hours_per_week`: Numbers of hours student studies per week
* `n_male`: Number of male
* `n_female`: Number of female 
* `number_of_siblings`: Number of siblings
* `sleep_time`: Daily Sleeping time (hours:mins)
* `wake_time`: Daily waking up time (hours:mins)
* `Age`: Age of the student

# Sypnopsis of the problem. 
* **Regression**: A simple regression problem on the final_test score. The test score exhibits normal distribution and the evalutation criteria will be based on Mean Absolute Error (MAE).

* **Classification**: The classification problem will classify all the grades into ordinal features ranging from 1-10. Since the classes are ordinal, it would be appropriate to evaluate them based on MAE of the respective classes. 


## Overview of Submitted folder
.
├── eda.ipnyb
├── __init__.py
├── data
│   └── score.db
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