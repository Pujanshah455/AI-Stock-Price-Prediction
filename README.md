# 📈 AI Stock Price Prediction

A Machine Learning-powered web app that predicts future stock prices using **LSTM Neural Networks**. Built with **TensorFlow**, **Keras**, **Streamlit**, and **yfinance**, this project fetches real-time stock data and visualizes predictions interactively.

![stock-prediction](https://img.shields.io/badge/MachineLearning-LSTM-blue) ![streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![python](https://img.shields.io/badge/Python-3.9%2B-yellow)

---

## 🚀 Features

- 📊 **Real-Time Data Fetching** using `yfinance`
- 🔮 **Stock Price Prediction** with LSTM models
- 🧠 Trained using **Keras** and **TensorFlow**
- 📉 Historical Data Visualization (Closing Price, MA100, MA200)
- 🧪 Model evaluation with **Train/Test split**
- 🌐 Interactive Web App using **Streamlit**

---

## 🛠️ Tech Stack

| Tool         | Purpose                          |
|--------------|----------------------------------|
| Python       | Programming Language             |
| TensorFlow   | Deep Learning Framework          |
| Keras        | Neural Network Modeling          |
| yfinance     | Real-time Stock Data Fetching    |
| scikit-learn | Data Preprocessing               |
| Streamlit    | Web App Frontend                 |
| Matplotlib   | Plotting & Visualization         |

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/ai-stock-price-predictor.git
   cd ai-stock-price-predictor
   
2.Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate   # On Windows

3.Install dependies:
pip install -r requirements.txt

4.Run the app:
streamlit run stock_predictor.py


🧠 Model Overview
Model Type: LSTM (Long Short-Term Memory)

Training Data: Historical stock price from Yahoo Finance

Inputs: 100-day closing price windows

Output: Next day’s closing price

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

