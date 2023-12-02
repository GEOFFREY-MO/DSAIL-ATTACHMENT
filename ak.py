# -*- coding: utf-8 -*-
"""ak.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VXRfVMIQ6UR42jeqJeKKO7hfXQG0bzXy
"""
!pip install streamlit
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
import streamlit as st

def data_analysis(data_path):
    # Check for empty file and invalid format before attempting to read the file
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please upload a valid dataset file.")
        return

    if not uploaded_file.name.endswith(".csv"):
        st.error("Invalid file format. Please upload a CSV file.")
        return

    # Load the dataset
    try:
        data = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        st.error(
            "The uploaded file is empty or not in CSV format. Please upload a valid dataset file."
        )
        return

    # Missing values analysis
    st.header("Missing Values Analysis")

    # Missing values count and percentage
    missing_values = data.isnull().sum()
    missing_values_percentage = (missing_values / len(data)) * 100
    st.write(f"Missing values count:\n{missing_values}")
    st.write(f"Missing values percentage:\n{missing_values_percentage}")

    # Missing value patterns
    pattern_detection_method = st.selectbox("Choose pattern detection method:", ["Row-wise", "Column-wise"])

    if pattern_detection_method == "Row-wise":
        # Identify rows with high missing value concentration
        high_missing_rows = data[data.isnull().sum(axis=1) > 0.5].index
        st.write(f"Rows with high missing value concentration: {high_missing_rows}")

    elif pattern_detection_method == "Column-wise":
        # Identify columns with high missing value concentration
        high_missing_columns = data.columns[data.isnull().sum(axis=0) > 0.5]
        st.write(f"Columns with high missing value concentration: {high_missing_columns}")

        # Check for relationships between missing values across variables
        correlation_matrix = data.corr()
        st.write("Correlation matrix:")
        st.write(correlation_matrix)

        # Identify pairs of variables with high negative correlation and high missing values
        potentially_related_variables = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                if correlation_matrix.iloc[i, j] < -0.7 and (missing_values_percentage.iloc[i] > 0.5 or missing_values_percentage.iloc[j] > 0.5):
                    potentially_related_variables.append((correlation_matrix.index[i], correlation_matrix.index[j]))

        if potentially_related_variables:
            st.write(f"Potentially related variables with high negative correlation and missing values:")
            st.write(potentially_related_variables)

    # Descriptive statistics
    st.write("Descriptive statistics:")
    st.write(data.describe())

    # Correlation analysis
    correlation_matrix = data.corr()
    st.write("Correlation matrix:")
    st.write(correlation_matrix)

    # Find the strongest correlation
    strongest_correlation = correlation_matrix.max()
    correlated_variables = correlation_matrix.idxmax()
    st.write("Strongest correlation:", str(strongest_correlation), ".3f", "between", correlated_variables)

    # Make comments about the data
    if strongest_correlation.any() > 0.7:
        st.write("There is a strong positive correlation between the variables. This means that they tend to move in the same direction.")
    elif strongest_correlation < -0.7:
        st.write("There is a strong negative correlation between the variables. This means that they tend to move in opposite directions.")
    else:
        st.write("There is a weak correlation between the variables. This means that they do not tend to move in the same direction.")

    # Frequency distribution
    for column in data.columns:
        if pd.api.types.is_categorical_dtype(data[column]):
            st.write(f"Frequency distribution of {column}:")
            st.write(data[column].value_counts())

    return data  # Return the 'data' object


# Main Streamlit app
st.title("Data Analysis App")

# Upload the dataset
uploaded_file = st.file_uploader("Choose a dataset file")

# If a dataset is uploaded, perform the data analysis and store the data
if uploaded_file is not None:
    data = data_analysis(uploaded_file.name)

    # Visualization area
    st.sidebar.header("Visualization Area")

    # Select graph type
    graph_type = st.sidebar.selectbox("Select Graph Type", ["Scatter Plot", "Bar Chart", "Line Chart"])

    # Select x-axis and y-axis variables
    x_variable = st.sidebar.selectbox("Select X-axis Variable", data.columns, key='x')
    y_variable = st.sidebar.selectbox("Select Y-axis Variable", data.columns, key='y')

    # Create and display the selected graph
    st.subheader(f"{graph_type} - {x_variable} vs. {y_variable}")

    if graph_type == "Scatter Plot":
        fig = px.scatter(data, x=x_variable, y=y_variable)
    elif graph_type == "Bar Chart":
        fig = px.bar(data, x=x_variable, y=y_variable)
    elif graph_type == "Line Chart":
        fig = px.line(data, x=x_variable, y=y_variable)

    st.plotly_chart(fig)
else:
    st.write("Please upload a dataset file to perform the data analysis.")
