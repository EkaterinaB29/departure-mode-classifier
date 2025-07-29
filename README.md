# Departure Mode Classifier

This project was developed as part of my internship and focuses on building a machine learning pipeline to classify the **departure mode** of shipping containers as either **TRUCK** or **TRAIN**. The project demonstrates practical use of feature engineering, model training, and evaluation using real-world logistics data.

---

## Project Files

| File                  | Description                                |
|-----------------------|--------------------------------------------|
| `model_training.py`   | Main script: trains and evaluates models   |
| `data5.tsv`           | Input dataset (tab-separated)              |
| `requirements.txt`    | List of dependencies                       |
| `rf_model.pkl`        | Trained RandomForest model (generated)     |
| `xgb_model.pkl`       | Trained XGBoost model (generated)          |

---

## How to Run the Project

### Option Run Locally

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
2.Place data.tsv in the same directory as the script

3.Run the script

 ```bash
   python model_training.py
```
Option Run in Google Colab
Open Google Colab
Upload the following files:

model_training.py

data.tsv


Install required libraries by running:

 ```python
    !pip install xgboost scikit-learn pandas matplotlib seaborn
    %run model_training.py
 ```


