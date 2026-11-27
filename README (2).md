# Forest Cover Type Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Notebook](https://img.shields.io/badge/Jupyter-Colab%20ready-F9AB00)

A complete machine learning pipeline that predicts the **type of forest cover** (7 classes, such as Spruce/Fir, Lodgepole Pine and Aspen) of a 30m x 30m land patch from terrain measurements. Built as an AIML mini project for B.Tech (AIDS).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Evaluation Approach](#evaluation-approach)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Authors](#authors)

## Overview

The goal is to classify forest cover type from cartographic features only (elevation, slope, distances to water, roads and fire points, hillshade, wilderness area and soil type). This is a **multiclass classification** problem with strongly **imbalanced classes**, so every model is compared with a "always predict the most common class" baseline and evaluated with both accuracy and **macro-F1**.

## Dataset

**UCI Covertype**, loaded directly with `sklearn.datasets.fetch_covtype` (no manual download needed).

| Property | Value |
|---|---|
| Full dataset | 581,012 rows |
| Used here | 12,000 rows (stratified random sample, so class shares are preserved) |
| Features | 54 (10 continuous, 4 wilderness-area flags, 40 soil-type flags) |
| Target | `Cover_Type`, 7 classes |
| Missing values | None |

## Project Pipeline

| Step | What is done |
|---|---|
| 1. EDA | Class balance, histograms, IQR outliers, correlations, per-class feature comparison, single-feature signal check |
| 2. Clustering | K-Means with elbow curve (automatic best k) and silhouette check; DBSCAN with k-distance plot for `eps` |
| 3. Classification | Logistic Regression, KNN, Decision Tree, Random Forest, Hist Gradient Boosting on all 54 features |
| 4. Feature selection | `VarianceThreshold`, `SelectKBest` (F-test and mutual information), `RFE`, `SelectFromModel` |
| 5. PCA | Scree plot, 95% variance, models trained on principal components |
| 6. Ensembles | Hard voting and soft voting (`VotingClassifier`) |
| 7. Final test | Best pipeline tested once on a held-out 20% test set, with classification report and confusion matrix |

## Evaluation Approach

- **Held-out test set (20%)**, stratified, untouched until the final step.
- **Stratified 5-Fold CV** on the training set for all model comparisons.
- **Pipelines** for scaling, feature selection and PCA, so they are fitted only on training folds (no data leakage).
- **Baseline** (`DummyClassifier`) shown next to every result.
- **Metrics:** accuracy, macro-F1, per-class report.
- Fixed random seed (`RANDOM_STATE = 42`) for reproducible results.

## How to Run

**Google Colab (recommended)**
1. Open `AIML_Project_Forest_Cover_Final.ipynb` in Colab.
2. Choose **Runtime > Run all**.
3. Internet is needed the first time to download the dataset (about 10 MB). The full run takes a few minutes on the free CPU.

**Locally**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook AIML_Project_Forest_Cover_Final.ipynb
```

## Repository Structure

```
.
├── AIML_Project_Forest_Cover_Final.ipynb   # full project notebook
└── README.md
```

## Tech Stack

Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn

## Results

All final numbers (baseline, per-model CV scores, feature selection and PCA verdicts, ensemble results, test-set accuracy and macro-F1, weakest class) are printed by code in **Section 11** of the notebook, so they always match the outputs of your run.

## Limitations and Future Work

- Only a 12,000-row sample (about 2% of the data) is used to keep Colab fast, and models use default hyperparameters.
- Rare classes (for example Cottonwood/Willow) have very few samples, so their scores vary between runs.
- Ideas: use more rows, tune with `GridSearchCV`, handle imbalance with `class_weight="balanced"` or SMOTE, and try weighted soft voting.

## Authors

| Name | Roll No. |
|---|---|
| Anurag Singh | 25AI1008 |
| Ajeet Kumar Yadav | 25AI1003 |

B.Tech (AIDS) | Subject: AIML | Mini Project
