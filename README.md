# Project : Panel Data Visualization

Student Performance Analysis

## Project Description

The purpose of this project is to answer a business question related to the dataset on student performance in exams. The project aims to explain and predict the math, reading and writing scores of high school students in the United States. 

The data application is designed to have two dashboards: 

1. Home Page: This will be the landing page from which all other dashboards will be accessible. 
2. Exploration Dashboard: This dashboard will allow users to explore the dataset using various filters (columns of the dataset). It will include plots such as bar plots, scatter plots, box plots, pie plots, etc. 
3. Analysis Dashboard: This dashboard will present the results of the analysis and the answer to the business question. Machine learning algorithms Linear Regression and  Clustering.


## Technologies used

The project will be implemented using the following technologies:

- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly go
- HoloViews
- Panel


## Dataset

The dataset for this project is the "Students Performance in Exams" dataset which contains information on student performance in math, reading and writing exams. The dataset contains information on various factors such as gender, race/ethnicity, parental level of education, lunch, test preparation course, etc.
link of the dataset : https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
  
## Installation
To install the necessary packages, run the following command:
pip install -r requirements.txt

## Usage

To view the website, run the following command
panel serve DataAnalysis.py ML.py --index=HomePage.html



