import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Set page title
st.set_page_config(page_title="Data Analysis Web App", layout="wide")

# Sidebar with options
st.sidebar.header("RESOURCE")
show_raw_data = st.sidebar.checkbox("Show Raw Data")
show_summary_statistics = st.sidebar.checkbox("Show Summary Statistics")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Main content area
selected_tab = st.sidebar.radio("### Navigation", ["Home", "Data Preprocessing", "Explore and Visualize", "Machine Learning"])

# Home tab
if selected_tab == "Home":
    st.header("Welcome to the Data Analysis Web App")

# Data Preprocessing tab
elif selected_tab == "Data Preprocessing":
    st.header("Data Preprocessing")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Display the raw data
            if show_raw_data:
                st.subheader("Raw Data")
                st.write(data)

            # Check data format
            if show_summary_statistics:
                st.subheader("Summary Statistics")
                st.write(data.describe())

            # Data preprocessing
            if st.checkbox('Clean and preprocess data'):
                st.subheader('Data Preprocessing')

                # Handling missing values
                handle_missing_values = st.checkbox('Handle missing values')
                if handle_missing_values:
                    st.write('Handling missing values...')
                    # allow user to choose a method for handling missing values
                    missing_method = st.selectbox(
                        'Select missing values handling method',
                        ['Mean', 'Median', 'Drop Rows']
                    )
                    if missing_method == 'Mean':
                        # Handle missing values for numeric columns
                        numeric_columns = data.select_dtypes(include="number").columns
                        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

                        # Handle missing values for non-numeric columns
                        non_numeric_columns = data.select_dtypes(exclude="number").columns
                        data[non_numeric_columns] = data[non_numeric_columns].fillna(data[non_numeric_columns].mode().iloc[0])
                    elif missing_method == 'Median':
                        data.fillna(data.median(), inplace=True)
                    elif missing_method == 'Drop Rows':
                        data.dropna(axis=0, inplace=True)
                    else:
                        st.warning('Please select a method for handling missing values.')
                    st.success('Missing values handled.')

                # Encoding categorical variables
                encode_categorical = st.checkbox('Encode categorical variables')
                if encode_categorical:
                    st.write('Encoding categorical variables...')
                    # allow users to choose columns for one-hot encoding
                    categorical_columns = st.multiselect(
                        'Select categorical columns for one-hot encoding', data.select_dtypes(include="object").columns
                    )
                    if categorical_columns:
                        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
                        st.success('Categorical variables encoded.')
                    else:
                        st.warning('No categorical columns selected for encoding.')

                # Scaling numerical features
                scale_numerical = st.checkbox("Scale numerical features")
                if scale_numerical:
                    st.write('Scaling numerical features...')
                    # allow users to choose columns for scaling
                    numerical_columns = st.multiselect(
                        'Select numerical columns for scaling', data.select_dtypes(include="number").columns
                    )
                    if numerical_columns:
                        # you can use different scaling method
                        data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / (
                                data[numerical_columns].max() - data[numerical_columns].min()
                        )
                        st.success('Numerical features scaled.')
                    else:
                        st.warning('No numerical columns selected for scaling.')

                # Store preprocessed data in session state
                st.session_state.data = data.copy()

                # Display the preprocessed data
                st.subheader('Preprocessed Data')
                st.write(st.session_state.data)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Explore and Visualize tab
elif selected_tab == "Explore and Visualize":
    st.header("Explore and Visualize Data")

    # Check if data is preprocessed
    if st.session_state.data is not None:
        data = st.session_state.data  # Use the preprocessed data for visualization

        try:
            plot_options = ["Seaborn", "Matplotlib", "Plotly"]
            selected_plot = st.selectbox("Select Visualization Tool", plot_options)

            if selected_plot == "Seaborn":
                st.write("### Seaborn Visualization")
                seaborn_plot_options = ['Histogram', 'Boxplot', 'Countplot']
                selected_seaborn_plot = st.selectbox('Select Seaborn Plot', seaborn_plot_options)

                if selected_seaborn_plot == 'Histogram':
                    st.write('## Histogram')
                    hist_column = st.selectbox('Select a column for the histogram', data.columns)
                    fig, ax = plt.subplots()
                    sns.histplot(data[hist_column], kde=True)
                    st.pyplot(fig)

                elif selected_seaborn_plot == 'Boxplot':
                    st.write('## Boxplot')
                    boxplot_x = st.selectbox('Select the X-axis column', data.columns)
                    boxplot_y = st.selectbox('Select the Y-axis column', data.columns)
                    fig, ax = plt.subplots()
                    sns.boxplot(x=boxplot_x, y=boxplot_y, data=data)
                    st.pyplot(fig)

                elif selected_seaborn_plot == 'Countplot':
                    st.write("## Countplot")
                    countplot_column = st.selectbox('Select a column for the countplot', data.columns)
                    fig, ax = plt.subplots()
                    sns.countplot(x=countplot_column, data=data)
                    st.pyplot(fig)

            elif selected_plot == "Matplotlib":
                st.write("### Matplotlib Visualization")
                matplotlib_plot_options = ['Line Plot', 'Scatter Plot', 'Bar Plot']
                selected_matplotlib_plot = st.selectbox('Select Matplotlib Plot', matplotlib_plot_options)

                if selected_matplotlib_plot == 'Line Plot':
                    st.write('### Line Plot')
                    lineplot_x = st.selectbox('Select the X-axis column', data.columns)
                    lineplot_y = st.selectbox('Select the Y-axis column', data.columns)
                    fig, ax = plt.subplots()
                    ax.plot(data[lineplot_x], data[lineplot_y])
                    ax.set_xlabel(lineplot_x)
                    ax.set_ylabel(lineplot_y)
                    st.pyplot(fig)

                elif selected_matplotlib_plot == "Scatter Plot":
                    st.write("### Scatter Plot")
                    scatterplot_x = st.selectbox("Select the X-axis column", data.columns)
                    scatterplot_y = st.selectbox("Select the Y-axis column", data.columns)
                    fig, ax = plt.subplots()
                    ax.scatter(data[scatterplot_x], data[scatterplot_y])
                    ax.set_xlabel(scatterplot_x)
                    ax.set_ylabel(scatterplot_y)
                    st.pyplot(fig)

                elif selected_matplotlib_plot == "Bar Plot":
                    st.write("### Bar Plot")
                    barplot_x = st.selectbox("Select the X-axis column", data.columns)
                    barplot_y = st.selectbox("Select the Y-axis column", data.columns)
                    fig, ax = plt.subplots()
                    ax.barh(data[barplot_x], data[barplot_y])  # Use barh for horizontal bar plot
                    ax.set_xlabel(barplot_x)
                    ax.set_ylabel(barplot_y)
                    st.pyplot(fig)

            # Plotly Visualization
            elif selected_plot == "Plotly":
                st.write("### Plotly Visualization")
                plotly_plot_options = ["Scatter Plot", "Line Plot", "Bar Plot"]
                selected_plotly_plot = st.selectbox("Select Plotly Plot", plotly_plot_options)

                if selected_plotly_plot == "Scatter Plot":
                    st.write("### Scatter Plot")
                    scatterplot_x = st.selectbox("Select the X-axis column", data.columns)
                    scatterplot_y = st.selectbox("Select the Y-axis column", data.columns)
                    st.plotly_chart(px.scatter(data, x=data[scatterplot_x], y=data[scatterplot_y], title="Scatter Plot"))

                elif selected_plotly_plot == "Bar Plot":
                    st.write("### Bar Plot")
                    barplot_x = st.selectbox("Select the X-axis column", data.columns)
                    barplot_y = st.selectbox("Select the Y-axis column", data.columns)
                    st.plotly_chart(px.bar(data, x=data[barplot_x], y=data[barplot_y], title="Bar Plot"))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please preprocess the data in the 'Data Preprocessing' tab first.")

# Machine Learning tab
elif selected_tab == "Machine Learning":
    st.header("Machine Learning")

    # Check if data is preprocessed
    if st.session_state.data is not None:
        data = st.session_state.data  # Use the preprocessed data for machine learning

        try:
            if st.checkbox("Build machine learning models"):
                target_variable = st.selectbox("Select the target variable", data.columns)
                features = data.drop(columns=[target_variable])

                # Identify categorical columns for one-hot encoding
                categorical_columns = data.select_dtypes(include="object").columns

                # One-hot encode categorical variables
                data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

                # Build selected model
                selected_model = st.selectbox("Select a machine learning model", ["Random Forest", "SVM", "Other Models"])
                if selected_model == "Random Forest":
                    model = RandomForestClassifier()
                    st.write("Random Forest model created.")

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, data[target_variable], test_size=0.2, random_state=42
                    )

                    # Build the selected model
                    model.fit(X_train, y_train)

                    # Make predictions on the test set
                    predictions = model.predict(X_test)

                    # Evaluate the model
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Accuracy: {accuracy:.2f}")

                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    st.write(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted']))

                    # Display precision, recall, and F1-score
                    st.subheader("Classification Report")
                    classification_report_str = classification_report(y_test, predictions)
                    st.text(classification_report_str)

                    # ROC Curve and AUC
                    st.subheader("Receiver Operating Characteristic (ROC) Curve")
                    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots()
                    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(fig)

                # Add more models as needed
                elif selected_model == "SVM":
                    # Support Vector Machine (SVM) model
                    model = SVC()  # Import the required class, SVC (Support Vector Classification)
                    st.write("SVM model created.")

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, data[target_variable], test_size=0.2, random_state=42
                    )

                    # Build the selected model
                    model.fit(X_train, y_train)

                    # Make predictions on the test set
                    predictions = model.predict(X_test)

                    # Evaluate the model
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Accuracy: {accuracy:.2f}")
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    st.write(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted']))

                    # Display precision, recall, and F1-score
                    st.subheader("Classification Report")
                    classification_report_str = classification_report(y_test, predictions)
                    st.text(classification_report_str)

                    # ROC Curve and AUC
                    st.subheader("Receiver Operating Characteristic (ROC) Curve")
                    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots()
                    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(fig)

                # Add more models as needed
                elif selected_model == "Other Models":
                    # Include code for other models
                    pass

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please preprocess the data in the 'Data Preprocessing' tab first.")
