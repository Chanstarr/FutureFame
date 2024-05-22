# FutureFame: Pioneering Student Success Using Various Regression Models

## Overview
The "FutureFame" project pioneers a data-driven approach to enhance student success prediction through the application of various regression models. By leveraging a rich dataset comprising demographic information, extracurricular activities, attendance records, test preparation efforts, university type, and past academic performance, the project aims to construct predictive models capable of estimating student scores accurately.

## Objectives
- To predict the student performance by analyzing various features with relevant data. 
- To minimize data noise and inconsistencies by preprocessing the collected dataset. 
- To build a pipeline that extracts relevant features from web data (CustomData). 
- To design a prediction system for student’s performance and novel clustering algorithm. 

## Project Structure

The project has the following structure:
   
```sh
└── FutureFame/
    ├── Models
    │   ├── ada_model.pkl
    │   ├── cat_model.pkl
    │   ├── dec_model.pkl
    │   ├── gra_model.pkl
    │   ├── lasso_model.pkl
    │   ├── lr_model.pkl
    │   ├── r_model.pkl
    │   ├── rnd_model.pkl
    │   └── xgb_model.pkl
    ├── Notebook
    │   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
    │   ├── 2. MODEL TRAINING.ipynb
    │   └── catboost_info
    │       ├── catboost_training.json
    │       ├── learn
    │       │   └── events.out.tfevents
    │       ├── learn_error.tsv
    │       └── time_left.tsv
    ├── app.py
    ├── catboost_info
    │   ├── catboost_training.json
    │   ├── learn
    │   │   └── events.out.tfevents
    │   ├── learn_error.tsv
    │   └── time_left.tsv
    ├── data
    │   ├── data.csv
    │   ├── test.csv
    │   └── train.csv
    ├── flask_session
    │   ├── 2029240f6d1128be89ddc32729463129
    │   └── cfbbab9984b7bef4ffd27256fe806024
    ├── preprocessor
    │   └── preprocessor.pkl
    ├── setup.py
    ├── src
    │   ├── components
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   └── model_trainer.py
    │   ├── exception.py
    │   ├── logger.py
    │   ├── pipeline
    │   │   └── predict_pipeline.py
    │   └── utils.py
    ├── static
    │   ├── analysis
    │   │   ├── accuracy.png
    │   │   ├── accuracy1.png
    │   │   ├── attendance.png
    │   │   ├── vs1.png
    │   │   └── vs2.png
    │   └── css
    │       ├── pred.css
    │       ├── styles.css
    │       └── styles_home.css
    ├── templates
    │   ├── analysis.html
    │   ├── front.html
    │   ├── home.html
    │   ├── index.html
    │   ├── login.html
    │   ├── prediction_results.html
    │   └── upload.html
    ├── uploads
    │   └── new.csv
    ├── user_info.txt
    └── users.txt
```

## Installation

1. Clone the repository: `git clone https://github.com/Chanstarr/FutureFame'
2. Navigate to the project directory: `cd FutureFame`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Run the application: `python app.py`
2. Access the web interface in your browser at `http://localhost:5000`
3. In the Home page you would be asked to Login, give the username: User1 and Password: Password1 or username: User2 and Password: Password2.
4. In the Project we have already added the dataset, also we can change the dataset by uploading a new dataset in the Upload page.
5. Fill in the student information and submit the form to obtain the predicted score, accuracy of the selected model and the Result(Pass/Fail).
6. In the Analysis page, we can view the Performance analysis for the Dataset.

## Authors

- Amrith
- Athish Raj Mohan
- Chandan M
- Chandan M V

## Acknowledgments

We would like to express our gratitude to our guide Prof. K N Prashanth Kumar, the project coordinators, peer reviewers, and the entire staff of the Computer Science Department at BIT for their support and guidance.


