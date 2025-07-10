import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np

# Load data
data = pd.read_csv("AI-Data.csv")

st.title("🎓 Student Performance Prediction Dashboard")

# =======================
# SECTION 1: Graph Plots
# =======================

st.header("📊 Data Visualization")

# Dropdown for graph type
graph_choice = st.selectbox("Choose a graph to display", [
    "Marks Class Count",
    "Semester-wise",
    "Gender-wise",
    "Nationality-wise",
    "Grade-wise",
    "Section-wise",
    "Topic-wise",
    "Stage-wise",
    "Absent Days-wise"
])

# Function to generate graphs
def plot_graph(choice):
    fig, ax = plt.subplots(figsize=(10, 6))
    if choice == "Marks Class Count":
        sns.countplot(x='Class', data=data, order=['L', 'M', 'H'], ax=ax)
    elif choice == "Semester-wise":
        sns.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Gender-wise":
        sns.countplot(x='gender', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Nationality-wise":
        sns.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Grade-wise":
        sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Section-wise":
        sns.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Topic-wise":
        sns.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Stage-wise":
        sns.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    elif choice == "Absent Days-wise":
        sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    return fig

if st.button("Show Graph"):
    st.pyplot(plot_graph(graph_choice))

# ============================
# SECTION 2: Model Training
# ============================

st.header("🤖 Model Training & Evaluation")

# Preprocessing
data = data.drop(columns=["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"])
u.shuffle(data)

# Label Encoding
for col in data.columns:
    if data[col].dtype == object:
        le = pp.LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Split
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
split_index = int(0.7 * len(data))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Models
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}

# Train & evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = m.accuracy_score(y_test, pred)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "report": m.classification_report(y_test, pred, output_dict=True)
    }

# Display accuracies
for name, result in results.items():
    st.subheader(f"🔹 {name}")
    st.write(f"**Accuracy:** {round(result['accuracy'], 3)}")
    st.json(result["report"])

# ===============================
# SECTION 3: Prediction Interface
# ===============================

st.header("🎯 Predict a Student's Performance")

# Input form
with st.form("prediction_form"):
    rai = st.number_input("Raised Hands", 0, 100)
    res = st.number_input("Visited Resources", 0, 100)
    dis = st.number_input("Number of Discussions", 0, 100)
    absc = st.selectbox("Student Absence Days", ["Under-7", "Above-7"])
    absc_val = 1 if absc == "Under-7" else 0

    submit = st.form_submit_button("Predict")

if submit:
    input_array = np.array([rai, res, dis, absc_val]).reshape(1, -1)

    class_map = {0: "H", 1: "M", 2: "L"}

    for name, result in results.items():
        pred = result["model"].predict(input_array)[0]
        pred_label = class_map.get(pred, str(pred))
        st.success(f"{name}: Predicted Class ➡️ {pred_label}")
