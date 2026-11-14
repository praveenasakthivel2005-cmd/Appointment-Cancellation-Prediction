# 💇 Appointment Cancellation Prediction using Machine Learning

## 📌 Project Overview

The **Appointment Cancellation Prediction** project is a Machine Learning-based application designed to predict whether a salon appointment will be **Cancelled or Completed** based on customer booking details.

The project uses a Chennai Salon appointment dataset and compares multiple Machine Learning algorithms to identify the best-performing model.

A Streamlit web application allows users to enter appointment details and get instant prediction results.

## 🎯 Project Objectives

- Predict salon appointment cancellations using Machine Learning.
- Analyse customer booking patterns.
- Compare multiple ML algorithms.
- Identify the best-performing classification model.
- Build an interactive prediction application using Streamlit.
- Help salon businesses improve appointment planning and reduce potential revenue loss.

## 📂 Dataset Information

**Dataset:** Chennai Salon Appointment Dataset (Synthetic)

- Total Records: 1,500
- Total Customers: 593
- Completed Appointments: 1,161
- Cancelled Appointments: 261
- No-show Appointments: 78

### Dataset Features

- Booking ID
- Customer ID
- Salon Name
- Area
- City
- Service
- Service Price
- Staff Member
- Booking Date
- Appointment Date
- Days in Advance
- Booking Channel
- Appointment Status
- Cancelled By
- Cancellation Reason

**Note:** No-show records are excluded from the initial binary classification model. The prediction target is Cancelled (1) or Completed (0).

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Saving | Joblib |
| Development | Jupyter Notebook |
| Web Application | Streamlit |

## 🤖 Machine Learning Algorithms

Seven classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)
7. Extra Trees Classifier

## 📊 Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- 5-Fold Cross Validation F1 Score

The model comparison results are available in `model_comparison.csv`.

## 📈 Model Comparison Results

The saved model comparison file reports the following scores:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 100% | 100% |
| Decision Tree | 100% | 100% |
| Random Forest | 100% | 100% |
| Gradient Boosting | 100% | 100% |
| KNN | 100% | 100% |
| SVM | 100% | 100% |
| Extra Trees | 100% | 100% |

**Note:** These are the results recorded in the current experiment using a synthetic dataset. They should not be interpreted as guaranteed real-world performance. Further validation with unseen real-world data is required.

## 🖥️ Streamlit Application Features

- Interactive appointment input form
- Booking Date and Appointment Date selection
- Area and City dropdowns
- Salon Name selection
- Service selection
- Staff Member selection
- Booking Channel selection
- Service Price input
- Appointment cancellation prediction
- Cancellation probability display
- Prediction result display
- Clear Prediction option

## 📸 Application Screenshot

![Appointment Cancellation Prediction Dashboard](Appointment%20Cancellation%20Prediction%20Dashboard.png)

## 📁 Project Structure

```text
Appointment-Cancellation-Prediction/
│
├── app.py
├── best_cancellation_model.pkl
├── model_comparison.csv
├── requirements.txt
│
├── dataset/
│   └── chennai_salon_appointments_synthetic.csv
│
├── salon_appointment_cancellation_prediction.ipynb
│
└── README.md
```

## ⚙️ Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project Folder

```bash
cd Appointment-Cancellation-Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit Application

```bash
python -m streamlit run app.py
```

### 5. Open the Application

Streamlit will open the application in your browser.

## 🔍 How It Works

1. Load the Chennai Salon appointment dataset.
2. Perform data cleaning and preprocessing.
3. Create the binary prediction target.
4. Split the dataset into training and testing sets.
5. Train and compare seven Machine Learning algorithms.
6. Evaluate model performance.
7. Save the selected model using Joblib.
8. Load the trained model into Streamlit.
9. Enter customer appointment details.
10. Display the prediction result.

## 🚀 Future Improvements

- Customer segmentation analysis.
- Advanced appointment analytics dashboard.
- Real-time database integration.
- Improved model validation using real-world data.
- Appointment reminder and notification system.

## 👩‍💻 Author

**G. Praveena**

Aspiring Data Analyst | Machine Learning Enthusiast

**Skills:** Python, SQL, Power BI, Machine Learning, Data Analysis

**LinkedIn:** [Connect with me](https://www.linkedin.com/in/praveena-sakthivel-2649842a4)

---

⭐ If you find this project useful, feel free to star this repository!
