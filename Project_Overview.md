# Data-Driven detection of Abnormal Electricity usage in Power supply Lines

---

## 1. Problem Statement
Electricity theft is a major issue in rural and urban India. People illegally tap distribution lines (pole wires) bypassing the meter, causing financial losses to electricity boards and honest consumers. The goal is to build an ML-based system that detects abnormal electricity consumption patterns automatically.

## 2. Dataset Description
File: electricity_theft_dataset.csv
Total Rows: 5000
Type: Synthetic — generated based on realistic electrical behavior

### Columns:
| Column | Type | Description |
|---|---|---|
| Timestamp | String | Date & time of reading (every 30 min) |
| Voltage_V | Float | Supply voltage in Volts (normal: 215-225V) |
| Current_A | Float | Current drawn in Amperes (normal: 0.4-0.9A) |
| Power_W | Float | Power consumption in Watts (V × I) |
| Power_Factor| Float | Electrical efficiency (normal: 0.88-0.96) |
| Season | String | Winter / Summer |
| Time_of_Day | String | Morning / Afternoon / Evening / Night |
| Delta_I | Float | Rate of change of current (sudden spike indicator) |
| Anomaly_Duration_min | Int | How long abnormal condition lasted (0 if normal) |
| Label | String | Normal / Abnormal (target variable) |

---

## 3. Methodology & Progress

### Phase 1: Exploratory Data Analysis (EDA)
- Analyzed class distribution: Found significant imbalance (85% Normal, 15% Abnormal).
- Identified key features: Delta_I and Power_Factor showed the strongest correlation with theft signatures.

### Phase 2: Data Preprocessing & Balancing
- Preprocessing: Label Encoding and StandardScaler were used for feature normalization.
- Handling Imbalance: Implemented SMOTE to balance the training data (50:50 ratio).

### Phase 4: Model Evaluation (Final Comparison)
We tested multiple models including Logistic Regression, Random Forest, XGBoost, LightGBM, and AdaBoost.

| Model (with SMOTE) | Accuracy | Precision | Recall (Theft) | F1-Score |
|---|---|---|---|---|
| Random Forest | 85.3% | 50.7% | 68.9% | 0.58 |
| XGBoost | 86.1% | 53.1% | 60.4% | 0.56 |
| **AdaBoost** | **79.9%** | **40.9%** | **76.4%** | **0.53** |

---

## 4. Visual Evaluation

### Model Comparison
![Model Comparison](model_comparison.png)
Analysis: While Random Forest and XGBoost have higher accuracy, **AdaBoost** significantly outperforms them in **Recall**, making it the most effective model for detecting electricity theft.

### Feature Importance (Best Model)
![Feature Importance](feature_importance.png)
Analysis: Delta_I remains the strongest indicator for theft detection across all models.

---

## 5. Final Conclusion
The project successfully improved the theft detection rate from an initial 34% (Logistic Regression) to **76.4% (AdaBoost)** using SMOTE and ensemble boosting techniques. Although there is a minor trade-off in precision, the high recall ensures that the majority of abnormal usage cases are flagged for investigation.

**Champion Model:** AdaBoost Classifier (Balanced with SMOTE)
