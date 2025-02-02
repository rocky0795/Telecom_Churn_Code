# 📡 Telecom Churn Prediction & Business Insights 🚀

## 📌 Project Overview

This project aims to **predict customer churn in the telecom industry** using machine learning. By analyzing customer behavior data (recharges, call usage, data consumption), we identify at-risk customers and recommend **strategies to reduce churn**.

## 🏆 Objectives

- Identify key **churn indicators** using customer behavior data.
- Train **machine learning models** to predict churn.
- Provide **business recommendations** to retain high-value customers.

## 📊 Dataset Overview

- **Rows:** 99,999 customer records
- **Columns:** Recharge amounts, call usage, data usage, roaming behavior over 4 months (June-September).
- **Target Variable:** **Churn** (1 = churned, 0 = retained)

## 🔎 Data Preprocessing Steps

✔ **Filtered high-value customers** (top 30% based on recharge).\
✔ **Dropped September data** (to avoid using churn-month data in training).\
✔ **Handled missing values & outliers** to improve model accuracy.\
✔ **Feature Engineering** created new predictive variables.

## 🔥 Feature Engineering Highlights

We created additional features to improve prediction:

- **Recharge Decay Rate:** Measures declining recharge behavior.
- **Data Usage Drop-off:** Captures sudden reductions in data consumption.
- **Last Recharge Gap:** Measures gaps between recharges.
- **Stickiness Score:** Identifies highly engaged users.

## 🤖 Model Training & Performance

We trained multiple machine learning models:

| Model                    | Accuracy | ROC-AUC Score |
| ------------------------ | -------- | ------------- |
| Logistic Regression      | 85%      | 0.82          |
| Random Forest            | 88%      | 0.89          |
| **XGBoost (Best Model)** | **92%**  | **0.92**      |

- **Best Model:** **XGBoost**
- **Handling Class Imbalance:** Used **undersampling** to balance churn vs. non-churn cases.

## 📊 Feature Importance Analysis (SHAP)

Top churn-driving features:
1️⃣ **date\_of\_last\_rech\_8** → Time since last recharge.\
2️⃣ **total\_ic\_mou\_8** → Total incoming call minutes.\
3️⃣ **loc\_ic\_mou\_8** → Local incoming call minutes.\
4️⃣ **last\_day\_rch\_amt\_8** → Last recharge amount.\
5️⃣ **roam\_og\_mou\_8** → Roaming outgoing minutes.

## 📌 Business Recommendations

| Feature                     | Churn Risk Insight                         | Business Action                                            |
| --------------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **date\_of\_last\_rech\_8** | Customers delaying recharges churn more    | Send **personalized reminders & discount offers**          |
| **total\_ic\_mou\_8**       | Fewer incoming calls → disengagement       | Introduce **"Refer & Earn" offers** to increase engagement |
| **roam\_og\_mou\_8**        | Customers using roaming are likely to stay | Provide **"Frequent Traveler" roaming packs**              |
| **last\_day\_rch\_amt\_8**  | Low last recharge amounts                  | Offer **loyalty recharge discounts** to increase spending  |

## 🚀 How to Use the Model

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the Jupyter Notebook

```
jupyter notebook
```

Open **Telecom\_Churn\_Prediction.ipynb** and execute all cells.

### 3️⃣ Predict Churn for a New Customer

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load("telecom_churn_model.pkl")

# Example customer data
new_customer = pd.DataFrame({
    "total_rech_amt_8": [500],
    "date_of_last_rech_8": [5],
    "total_ic_mou_8": [50],
    "roam_og_mou_8": [20]
})

# Predict churn
prediction = model.predict(new_customer)
print("Churn Prediction:", "Churn" if prediction[0] == 1 else "Not Churn")
```

## 📁 Project Structure

```
📂 Telecom-Churn-Prediction
│── 📄 README.md                <- Project documentation
│── 📄 Telecom_Churn_Prediction.ipynb  <- Jupyter Notebook with analysis & modeling
│── 📄 Telecom Churn Prediction & Business Recommendations.pdf
│── 📄 Capstone.zip <- Contains Final trained model
```

## 🎯 Future Improvements

- Include **customer feedback data** for deeper churn insights.
- Implement **real-time churn alerts** for proactive intervention.
- Use **deep learning models (LSTMs)** for better sequential pattern detection.

## 👨‍💻 Author

- **Rocky Ranjan**
- 📧 rocky.ranjan07\@gmail.com
