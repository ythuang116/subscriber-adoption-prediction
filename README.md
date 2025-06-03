# Subscriber Adoption Prediction

## Project Background

Website XYZ is a music-listening social networking platform operating on a “freemium” model: basic features are free, while premium capabilities require a monthly subscription. In our last marketing campaign, 41,540 free users were targeted, of which 1,540 converted to premium subscribers (the adopters), and the remaining 40,000 did not.

Our objective is to predict which users are most likely to convert to premium within the next six months if they receive a targeted promotional offer. This problem poses a significant class-imbalance challenge, as only about 3.7% of users became adopters.

## Dataset Details

| Metric        | Value                  |
| ------------- | ---------------------- |
| Total records | 41,540 users           |
| Adopters      | 1,540 (≈3.7%)          |
| Non-adopters  | 40,000                 |
| Features      | 25 attributes per user |

## Pipeline Overview

The full workflow is implemented in `notebooks/XYZ_prediction.ipynb` and summarized below:

1. **Exploratory Data Analysis (EDA)**

   - **Data Loading & Inspection**: Read 41,540 records; check data types and missing values.
   - **Class Imbalance Investigation**: Visualize adopter vs. non-adopter counts; calculate imbalance ratio.
   - **Feature Distributions & Outliers**: Plot histograms and boxplots for key variables (e.g., friend count, track plays).
   - **Correlation Analysis**: Examine feature-target relationships to identify strong predictors of conversion.

2. **Preprocessing**

   - **Feature Engineering**: Derive engagement ratios (e.g., plays per month), encode categorical indicators.
   - **Scaling & Split**: Normalize numeric features; split data into training (80%) and testing (20%) sets.
   - **Resampling**: Use **SMOTE** to oversample the minority class in the training set, mitigating bias.

3. **Model Training & Evaluation**

   - **Baseline Model**: Logistic Regression to establish a reference point under imbalance.
   - **Ensemble Methods**: Train Random Forest and **LightGBM** classifiers with hyperparameter tuning via **GridSearchCV**.
   - **Cross-Validation**: Perform stratified 3-fold CV to ensure robust performance estimates.

4. **Performance Metrics**

   - **Classification Report**: Precision, recall, F1-score for adopters and non-adopters.
   - **ROC AUC Score**: Measure overall discrimination ability.
   - **Cost-Sensitive Analysis**: Define and apply custom costs for false positives (FP) and false negatives (FN) to reflect business impact; compute and report total cost based on the confusion matrix.
   - **Visualizations**: Confusion matrices and ROC curves saved in `results/figures/`.

5. **Results Summary**

| Model               | Precision (Adopter) | Recall (Adopter) | F1-score (Adopter) | ROC AUC | Total Cost |
| ------------------- | ------------------- | ---------------- | ------------------ | ------- | ---------- |
| Logistic Regression | 0.12                | 0.36             | 0.18               | 0.78    | $2791      |
| Random Forest       | 0.12                | 0.44             | 0.19               | 0.80    | $2733      |
| LightGBM            | 0.12                | 0.45             | 0.19               | 0.80    | $2701      |

6. **Final Deliverables**
   - **Notebook**: `notebooks/XYZ_prediction.ipynb` with full code, commentary, and plots.
   - **Documentation**: This `README.md` and a summary report in `docs/Report.pdf`.

