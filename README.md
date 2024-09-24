# Design-Evaluation---Kmeans-Clustering
Machine learning analysis of problem statements using clustering and agreement metrics.

This repository contains code for evaluating 177 problem statements using a 2x2 design dimension scale. The project applies K-Means clustering to analyze the relationship, and employs Intraclass Correlation Coefficient (ICC) analysis to measure inter-rater agreement.

# Methods Used
- K-Means Clustering
- Intraclass Correlation Coefficient (ICC)
- Scatter Plot Visualizations

# Project Setup
1. Clone the repository:  
 git clone https://github.com/Problem-Evaluation-Analysis.git

2. Install dependencies:  
pip install -r requirements.txt

3. Run the models on the dataset:
- python data_preprocessing.py for preprocessing data and calculating z-scores.
- python kmeans_clustering.py for K-Means clustering analysis.
- python icc_analysis.py for ICC inter-rater agreement analysis.

# Tech Stack
Python
Libraries:
- scikit-learn (for clustering)
- pandas (for data manipulation)
- matplotlib and seaborn (for visualizations)
- pingouin (for ICC analysis)
