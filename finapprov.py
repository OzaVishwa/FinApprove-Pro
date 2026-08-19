import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

#step 1 read csv file
df = pd.read_csv("FinApprovePro_dataset.csv")

with st.sidebar:
    selected = option_menu("Finapprov Pro", options = ["DASHBOARD", "DATASET", "EDA", "PREDICTION", "STATISTICS", "PROJECT DESCRIPTION",
                                                                   "PROFILE"], menu_icon=["cloud"],icons = ["house",
                                                                    "table", "graph-up", "search", "info-circle", "envelope", "person"])

if selected == "DASHBOARD":
        st.header(" FinApprove Pro")
        st.subheader("Intelligent Banking Decision Platform")

        st.image("Banking.jpg" ,
                  use_container_width=True)
        st.write("""
        Welcome to **FinApprove Pro**!

        This application uses Machine Learning to predict loan approval status
        and provides interactive analytics for banking data.
        """)

        # ---------------- KPI Cards ---------------- #

        total_records = len(df)
        avg_credit_score = round(df["credit_score"].mean(), 2)
        avg_loan_amount = round(df["loan_amount"].mean(), 2)
        avg_income = round(df["annual_income"].mean(), 2)

        approved = df["loan_approved"].sum()
        approval_rate = round((approved / total_records) * 100, 2)

        col1, col2, = st.columns(2)

        col1.metric("Total Records", total_records)
        col2.metric("Avg Credit Score", avg_credit_score)

        col3, col4 =st.columns(2)
        col3.metric("Avg Loan Amount", avg_loan_amount)
        col4.metric("Avg Annual Income", avg_income)
           
if selected=="DATASET":
    st.subheader("Dataset Overview: ")
    st.dataframe(df)

    st.subheader("Dataset Shape Information: ")
    st.write("Rows: ", df.shape[0])
    st.write("Columns: ", df.shape[1])
    st.subheader("Dataset Summary")
    st.write(df.describe())

if selected=="EDA":
    st.subheader("Explanatory Data Analysis: ")

    total_records = len(df)
    avg_credit_score = round(df["credit_score"].mean(), 2)
    type_of_loans = df["loan_purpose"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total_Records: ", total_records)
    col2.metric("Average_Credit_Score: ", avg_credit_score)
    col3.metric("Type_of_Loans: ",  type_of_loans)

    col4, col5 = st.columns(2)
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Loan Purpose Count")
        status = df["loan_purpose"].value_counts()
        st.write(status)

    with col5:
        st.subheader("Loan Purpose Visualization")
        plt.figure(figsize=(6, 4))
        plt.bar(status.index, status.values, color="violet")
        plt.xlabel("Loan Purpose")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    col6, col7 = st.columns(2)

    with col6:
        st.subheader("Employment Type Count")
        status1 = df["employment_type"].value_counts()
        st.write(status1)

    with col7:
        st.subheader("Employment Type Visualization")
        plt.figure(figsize=(6, 4))
        plt.bar(status1.index, status1.values, color="lightgreen")
        plt.xlabel("Employment Type")
        plt.ylabel("Count")
        st.pyplot(plt)

    col8, col9 = st.columns(2)

    with col8:
        st.subheader("Credit Score Statistics")
        st.write(df["credit_score"].describe())

    with col9:
        st.subheader("Credit Score Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["credit_score"], bins=10, color="coral", edgecolor="white")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Number of Applicants")
        ax.set_title("Distribution of Credit Scores")

        st.pyplot(fig)

    col10, col11 = st.columns(2)

    with col10:
        st.subheader("Loan Approval Distribution")

        approval = df["loan_approved"].value_counts()

        plt.figure(figsize=(4,4))
        plt.pie(
        approval.values,
        labels=["Approved", "Rejected"],
        autopct="%1.1f%%"
    )
        plt.title("Loan Approval")
        st.pyplot(plt)

    with col11:
        st.subheader("Credit Score vs Loan Amount")
        plt.figure(figsize=(5, 4))
        plt.scatter(
        df["credit_score"],
        df["loan_amount"],
        color="cyan",
        alpha=0.7,
    )
        plt.xlabel("Credit Score")
        plt.ylabel("Loan Amount")
        plt.title("Credit Score vs Loan Amount")
        st.pyplot(plt)

    col12, col13 = st.columns(2)
    with col12:
        st.subheader("Average Loan Amount by Employment Years")

        line_data = df.groupby("employment_years")["loan_amount"].mean()

        plt.figure(figsize=(6, 4))
        plt.plot(line_data.index, line_data.values, marker='o', color='green')

        plt.xlabel("Employment Years")
        plt.ylabel("Average Loan Amount")
        plt.title("Employment Years vs Average Loan Amount")

        st.pyplot(plt)

    with col13:
        st.subheader("Average Loan Amount by Existing Loans")

        line_data = df.groupby("existing_loans")["loan_amount"].mean()

        plt.figure(figsize=(6, 4))
        plt.plot(line_data.index, line_data.values, marker='o', color='deeppink')

        plt.xlabel("Existing Loans")
        plt.ylabel("Average Loan Amount")
        plt.title("Existing Loans vs Average Loan Amount")

        st.pyplot(plt)

if selected == "PREDICTION":
    le = LabelEncoder()
    le1 = LabelEncoder()

    df["employment_type"] = le.fit_transform(df["employment_type"])
    df["loan_purpose"] = le1.fit_transform(df["loan_purpose"])

    x = df[["annual_income","credit_score","employment_years","existing_loans",
            "loan_amount","employment_type","loan_purpose"]]
    y = df["loan_approved"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2,stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(xtrain, ytrain)

    prediction = model.predict(xtest)

    accuracy = accuracy_score(ytest, prediction)

    st.header("Loan Approval Prediction using Random Forest Classifier")
    st.subheader("Kindly Enter Value to get Prediction")

    Annual_income = st.number_input("Enter Your Annual Income: ", value=1000000)
    Credit_score = st.number_input("Enter Your Credit Score: ", value=250)
    Employment_yrs = st.number_input("Enter Employment Years: ", value=8)
    Existing_loans = st.number_input("Enter Num of Existing Loans: ", value=2)
    Loan_amt = st.number_input("Enter Desired Loan Amount: ", value=40000)
    Employment_type = st.selectbox(
        "Employment Type",
        le.classes_
    )

    Loan_purpose = st.selectbox(
        "Loan Purpose",
        le1.classes_
    )
    Employment_type = le.transform([Employment_type])[0]
    Loan_purpose = le1.transform([Loan_purpose])[0]

    if st.button("Predict Loan Status"):
        new_prediction = model.predict([[Annual_income,Credit_score,Employment_yrs,Existing_loans,
                                         Loan_amt,Employment_type,Loan_purpose]])
        print("new_prediction", le.inverse_transform(new_prediction))
        print("new_prediction", le1.inverse_transform(new_prediction))
        if new_prediction[0] == 1:
            st.success("Loan Approved")
            st.snow()
        else:
            st.error("😔 Loan Rejected")
            st.warning("💔 Unfortunately, your application could not be approved.")
        #st.success(f"Prediction: {new_prediction[0]}")
        st.info(f"Accuracy: {accuracy*100:.2f}%")

if selected == "STATISTICS":

    st.header(" Statistical Analysis ")

    # Correlations
    st.subheader("Correlation Matrix")
    st.dataframe(df.corr(numeric_only=True).round(2))

    st.markdown("---")

    # Loan Approval Statistics
    st.subheader("Loan Approval Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Approved Loans", (df["loan_approved"] == 1).sum())
        st.metric("Rejected Loans", (df["loan_approved"] == 0).sum())

    with col2:
        approval_rate = (df["loan_approved"] == 1).mean() * 100
        rejection_rate = (df["loan_approved"] == 0).mean() * 100

        st.metric("Approval Rate", f"{approval_rate:.2f}%")
        st.metric("Rejection Rate", f"{rejection_rate:.2f}%")

    st.markdown("---")

    # Credit Score Statistics
    st.subheader("Credit Score Statistics")

    col3, col4 = st.columns(2)

    with col3:
        st.write("Average Credit Score :", round(df["credit_score"].mean(), 2))
        st.write("Maximum Credit Score :", df["credit_score"].max())
        st.write("Minimum Credit Score :", df["credit_score"].min())

    with col4:
        st.write("Median Credit Score :", df["credit_score"].median())
        st.write("Standard Deviation :", round(df["credit_score"].std(), 2))
        st.write("Variance :", round(df["credit_score"].var(), 2))

    st.markdown("---")

    # Loan Statistics
    col5, col6 =st.columns(2)
    with col5:
     st.subheader("Loan Amount Statistics")

     st.write("Average Loan Amount :", round(df["loan_amount"].mean(), 2))
     st.write("Maximum Loan Amount :", df["loan_amount"].max())
     st.write("Minimum Loan Amount :", df["loan_amount"].min())

    # Income Statistics
    with col6:
     st.subheader("Annual Income Statistics")

     st.write("Average Annual Income :", round(df["annual_income"].mean(), 2))
     st.write("Highest Annual Income :", df["annual_income"].max())
     st.write("Lowest Annual Income :", df["annual_income"].min())

if selected == "PROJECT DESCRIPTION":
        st.header("◎ Project Description")

        st.markdown("""
        ## ▤ FinApprove Pro – Intelligent Banking Decision Platform

        **FinApprove Pro** is a Machine Learning-based web application developed using **Python** and **Streamlit** to predict whether a customer's loan application is likely to be approved or rejected. The application is designed to assist banks and financial institutions in making faster, smarter, and more reliable lending decisions by analyzing customer financial information. It provides an interactive dashboard where users can explore the dataset, visualize important trends, and predict loan approval status based on customer details. The project combines data analysis, visualization, and machine learning into a single user-friendly platform.

       ### ✧ Objectives
        - Predict loan approval using Machine Learning techniques.
        - Assist banks in making faster and more accurate loan approval decisions.
        - Reduce manual verification and processing time.
        - Provide an interactive dashboard for data exploration and visualization.
        - Improve decision-making through data-driven insights.
        - Enhance transparency in the loan approval process.

        ### ✧ Features
        - Dashboard displaying project overview and key information.
        - Dataset exploration with tabular view.
        - Exploratory Data Analysis (EDA) using different visualizations.
        - Loan Approval Prediction using Random Forest Classifier.
        - Statistical analysis of applicant data.
        - Interactive graphs including bar charts, pie charts, histograms, scatter plots, line charts, and box plots.
        - User-friendly interface developed using Streamlit.

        ### ✧ Technologies Used
        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Matplotlib
        - Scikit-learn
        - Random Forest Classifier

        ### ✧ Machine Learning Algorithm
        The project uses the **Random Forest Classifier**, an ensemble machine learning algorithm that builds multiple decision trees and combines their predictions to produce highly accurate and reliable results. Random Forest helps reduce overfitting, handles large datasets efficiently, and performs well with both numerical and categorical data, making it suitable for loan approval prediction.

        ### ✧ Input Parameters
        - Annual Income
        - Credit Score
        - Employment Years
        - Existing Loans
        - Loan Amount
        - Employment Type
        - Loan Purpose

        ### ✧ Output
        - Loan Approved
        - Loan Rejected

        ### ✧ Benefits
        - Faster loan approval process.
        - Improved prediction accuracy using Machine Learning.
        - Reduces manual effort and paperwork.
        - Supports banking professionals in decision-making.
        - Provides meaningful data visualizations for analysis.
        - Easy-to-use and interactive interface.
        - Helps financial institutions improve operational efficiency.

        ### ✧ Future Enhancements
        - Integration with real-time banking databases.
        - Support for multiple Machine Learning algorithms.
        - Advanced analytics dashboard with live reports.
        - Customer authentication and secure login.
        - Cloud deployment for online access.
        - Automated report generation and loan history tracking.
        """)

if selected == "PROFILE":

    st.header("👤 Developer Profile")

    st.markdown("""
### 👤 Personal Details

**Name:** Oza Vishwa A

**Course:** Information Technology (IT)

**College:** SAL Engineering & Technical Institute (SETI)

**Location:** Science City, Ahmedabad, Gujarat

**Email:** ozavishwa@zohomail.in

---

### ► Technical Skills

- Python
- Machine Learning
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

### ► Career Objective

To build intelligent and user-friendly Machine Learning applications while enhancing my skills in Data Science and Artificial Intelligence.

---

### ► About FinApprove Pro

FinApprove Pro is a Machine Learning-based banking application developed using Python and Streamlit. It predicts whether a customer's loan application will be approved or rejected using the Random Forest Classifier. The application includes an interactive dashboard, dataset exploration, exploratory data analysis (EDA), statistical insights, and various visualizations to assist banks in making faster, smarter, and more reliable loan approval decisions.

---
""")