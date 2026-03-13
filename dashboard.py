import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("AI Employee Attrition Dashboard")

# Load Dataset
df = pd.read_csv("HR-Employee-Attrition.csv")

# Convert Attrition
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

# Drop unnecessary columns
df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Features and target
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train,y_train)

# -----------------------------
# Employee Prediction Section
# -----------------------------

st.header("Employee Attrition Prediction")

emp_id = st.number_input("Enter Employee Number", min_value=1)

if st.button("Check Employee"):

    emp = X[X['EmployeeNumber']==emp_id]

    if emp.empty:
        st.write("Employee not found")

    else:
        prediction = model.predict(emp)
        probability = model.predict_proba(emp)

        risk = probability[0][1]*100

        st.write("Risk Score:", round(risk,2), "%")

        if prediction[0]==1:
            st.error("Employee likely to LEAVE")

        else:
            st.success("Employee likely to STAY")

# -----------------------------
# Visualization Section
# -----------------------------

st.header("Attrition Analysis")

fig1 = plt.figure()

sns.countplot(x='Attrition', data=df)

plt.title("Employee Attrition Count")

st.pyplot(fig1)


fig2 = plt.figure()

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)

plt.title("Salary vs Attrition")

st.pyplot(fig2)


fig3 = plt.figure()

sns.countplot(x='OverTime', hue='Attrition', data=df)

plt.title("Overtime vs Attrition")

st.pyplot(fig3)

# -----------------------------
# Top 5 Risk Employees
# -----------------------------

st.header("Top Employees Likely to Leave")

probabilities = model.predict_proba(X)

risk_scores = probabilities[:,1]

df['RiskScore'] = risk_scores

top5 = df.sort_values(by='RiskScore', ascending=False).head(5)

st.dataframe(top5[['EmployeeNumber','RiskScore']])