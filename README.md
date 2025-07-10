# 🎓 Student-Evaluator


An interactive web application that predicts student academic performance using multiple machine learning models. Built with Streamlit, this tool enables real-time grade predictions and provides data visualizations based on various educational features like attendance, gender, semester, and more.

---

## 📌 Features

- 📊 Visualizations: Generate interactive graphs on performance by gender, semester, grade level, and other features.
- 🤖 ML Models: Compares accuracy of multiple classifiers including:
  - Decision Tree
  - Random Forest
  - Perceptron
  - Logistic Regression
  - Neural Network (MLP Classifier)
- 🧠 Live Prediction: Allows users to input student data and receive real-time grade predictions from each model.
- 🖥️ User-Friendly UI: Streamlit-based interface for easy interaction and visualization.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Backend/ML**: Python, Scikit-learn  
- **Data Manipulation & Viz**: Pandas, NumPy, Seaborn, Matplotlib

---

## 📁 Dataset

- The dataset used: `AI-Data.csv`
- Contains features like:
  - Gender, Nationality, Grade, Section
  - Raised hands, Visited resources, Absence days
  - Final Grade Classification (`L`, `M`, `H`)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/StudentPerformancePrediction-ML.git
   cd StudentPerformancePrediction-ML
