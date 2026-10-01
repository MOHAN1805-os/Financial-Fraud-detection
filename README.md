# Financial Fraud Detection System

## Project Description

Financial Fraud Detection is a machine learning project that detects potentially fraudulent financial transactions.

The system uses a Random Forest classification model to analyze transaction data and classify transactions as fraudulent or genuine.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Machine Learning

## Features

* Transaction data preprocessing
* Fraud detection using Random Forest
* Model evaluation
* Fraud probability prediction
* CSV batch prediction
* Interactive Streamlit dashboard
* Fraud analysis charts
* Download prediction results

## Project Structure

```text
Financial-Fraud-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── fraud_data.csv
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

## Machine Learning Model

Random Forest Classifier is used for fraud classification.

The target variable is:

```text
isFraud
```

The model predicts:

```text
0 = Genuine Transaction
1 = Fraudulent Transaction
```

## Future Enhancements

* Real-time fraud detection
* Advanced machine learning models
* Cloud deployment
* Real-time transaction monitoring
* Email/SMS fraud alerts
* Improved dashboard analytics
