# Predict Which Water Pumps Are Faulty Using Reproducible Code

# Installation
The libraries/packages used are:

- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- logging

# Project Motivation
The project is the capstone project for the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) - Term2. 

The dataset of this project comes from [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) hosted by [DrivenData](https://www.drivendata.org/).

The goal of this project is to predict which pumps are functional, which need some repairs, and which don't work at all by using the data from [Taarifa](http://taarifa.org/) and the [Tanzanian Ministry of Water](http://maji.go.tz/). Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. It is important to make the code usable, reusable, maintainable and reproducible so that the work can be shared and used by other people. Understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania. [Source](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)

# File Descriptions
- This notebook `Predict Which Water Pumps Are Faulty Using Reproducible Code.ipynb` includes the analysis and the modeling steps which follows Cross-Industry Standard Process for Data Mining (CRISP-DM).
- The python file `rf_model.py` is the modularized code that predicts the label of the input data and evaluates model performance.
- The dataset can be downloaded from the [Driven Data Website](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/).

# Results
A Medium blog post that discuss the project in details is available [here](https://medium.com/@sophiaheli/predict-which-water-pumps-are-faulty-using-reproducible-code-aa42b3fc0320).

# Licensing, Authors, Acknowledgements
1. The project dataset is credit to [Harnessing the Power of the Crowd to Increase Capacity for Data Science in the Social Sector paper](https://arxiv.org/abs/1606.07781).
2. Credits to https://github.com/paawan01/Titanic_dataset_analysis for the readme template
3. I used below references when completing the project
    - [Datacamp course: Driven Data Water Pumps Challenge](https://www.datacamp.com/community/open-courses/drivendata-water-pumps-challenge)
    - [Open Data Science Conference 2018 East: Turning a Data Science Brain Dump into Software by Katie Malone, Ph.D.](https://github.com/cmmalone)

