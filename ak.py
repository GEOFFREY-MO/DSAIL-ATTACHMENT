# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page title
st.set_page_config(page_title="Data Analysis Web App", layout="wide")

# Sidebar with tabs
selected_tab = st.sidebar.radio("Navigation", ["Home", "Data Preprocessing", "Exploration and Visualization", "Machine Learning"])

# Home tab
if selected_tab == "Home":
    st.title('Data Analysis Web App')
    st.subheader('Enjoy data analysis without coding and create machine learning models easily with this Giant!')
    uploaded_file = st.file_uploader("Upload autodata CSV file", type=["csv"])

    # Sidebar with options
    st.sidebar.header("RESOURCE")
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    show_summary_statistics = st.sidebar.checkbox("Show Summary Statistics")

    # Main content area
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        if show_raw_data:
            st.subheader("Raw Data")
            st.write(data)

        # Check data format
        if show_summary_statistics:
            st.subheader("Summary Statistics")
            st.write(data.describe())

# Data Preprocessing tab
elif selected_tab == "Data Preprocessing":
    st.header("Data Preprocessing")
    st.sidebar.header("RESOURCE")
    uploaded_file = st.file_uploader("Upload autodata CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Data preprocessing
        if st.checkbox('Clean and preprocess data'):
            st.subheader('Data Preprocessing')

            # Handling missing values
            handle_missing_values = st.checkbox('Handle missing values')
            if handle_missing_values:
                st.write('Handling missing values...')
                # Allow user to choose a method for handling missing values
                missing_method = st.selectbox(
                    'Select missing values handling method',
                    ['Mean', 'Median', 'Drop Rows']
                )
                if missing_method == 'Mean':
                    data.fillna(data.mean(), inplace=True)
                elif missing_method == 'Median':
                    data.fillna(data.median(), inplace=True)
                elif missing_method == 'Drop Rows':
                    data.dropna(axis=0, inplace=True)

                st.success('Missing values handled.')

            # Encoding categorical variables
            encode_categorical = st.checkbox('Encode categorical variables')
            if encode_categorical:
                st.write('Encoding categorical variables...')
                # Allow users to choose columns for one-hot encoding
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
                # Allow users to choose columns for scaling
                numerical_columns = st.multiselect(
                    'Select numerical columns for scaling', data.select_dtypes(include="number").columns
                )
                if numerical_columns:
                    # You can use different scaling methods
                    data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / (
                            data[numerical_columns].max() - data[numerical_columns].min()
                    )
                    st.success('Numerical features scaled.')
                else:
                    st.warning('No numerical columns selected for scaling.')

            # Display the preprocessed data
            st.subheader('Preprocessed Data')
            st.write(data.head())

# Exploration and Visualization tab
elif selected_tab == "Exploration and Visualization":
    st.header("Explore and Visualize Data")
    st.sidebar.header("RESOURCE")
    uploaded_file = st.file_uploader("Upload autodata CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Explore and visualize data
        st.subheader("Explore and Visualize Data")
        plot_options = ["Seaborn", "Matplotlib", "Plotly"]
        selected_plot = st.selectbox("Select Visualization Tool", plot_options)

        if selected_plot == "Seaborn":
            st.write("### Seaborn Visualization")
            # dynamic dropdown for seaborn plot selection
            seaborn_plot_options = ['Histogram', 'Boxplot', 'Countplot']
            selected_seaborn_plot = st.selectbox('Select Seaborn Plot', seaborn_plot_options)
            # Seaborn code based on user's selection
            if selected_seaborn_plot == 'Histogram':
                st.write('## Histogram')
                # allow users to choose the column for the histogram
                hist_column = st.selectbox('Select a column for the histogram', data.columns)
                sns.histplot(data[hist_column], kde=True)
                st.pyplot()

            elif selected_seaborn_plot == 'Boxplot':
                st.write('## Boxplot')
                # allow user to choose the x and y axes for the boxplot
                boxplot_x = st.selectbox('Select the X-axis column', data.columns)
                boxplot_y = st.selectbox('Select the Y-axis column', data.columns)
                sns.boxplot(x=boxplot_x, y=boxplot_y, data=data)
                st.pyplot()
            elif selected_seaborn_plot == 'Countplot':
                st.write("## Countplot")
                # allow user to choose the column for the countplot
                countplot_column = st.selectbox('Select a column for the countplot', data.columns)
                sns.countplot(x=countplot_column, data=data)
                st.pyplot()

        elif selected_plot == "Matplotlib":
            st.write("### Matplotlib Visualization")
            # dynamic dropdown for matplotlib plot selection
            matplotlib_plot_options = ['Line Plot', 'Scatter Plot', 'Bar plot']
            selected_matplotlib_plot = st.selectedbox('Select Matplotlib Plot', matplotlib_plot_options)
            # matplotlib code based on user's selection
            if selected_matplotlib_plot == 'Line Plot':
                st.write('### Line Plot')
                # allow user to choose the x and y axes for the line plot
                lineplot_x = st.selectbox('Select the X-axis column', data.columns)
                lineplot_y = st.selectbox('Select the Y-axis column', data.columns)
                plt.plot(data[lineplot_x], data[lineplot_y])
                plt.xlabel(lineplot_x)
                plt.ylabel(lineplot_x)
                st.plotly()
            elif selected_matplotlib_plot == "Scatter Plot":
                st.write("### Scatter Plot")
                # Allow users to choose the x and y axes for the scatter plot
                scatterplot_x = st.selectbox("Select the X-axis column", data.columns)
                scatterplot_y = st.selectbox("Select the Y-axis column", data.columns)
                plt.scatter(data[scatterplot_x], data[scatterplot_y])
                plt.xlabel(scatterplot_x)
                plt.ylabel(scatterplot_y)
                st.pyplot()

            elif selected_matplotlib_plot == "Bar Plot":
                st.write("### Bar Plot")
                # Allow users to choose the x and y axes for the bar plot
                barplot_x = st.selectbox("Select the X-axis column", data.columns)
                barplot_y = st.selectbox("Select the Y-axis column", data.columns)
                plt.bar(data[barplot_x], data[barplot_y])
                plt.xlabel(barplot_x)
                plt.ylabel(barplot_y)
                st.pyplot()

        # Plotly Visualization
        elif selected_plot == "Plotly":
            st.write("### Plotly Visualization")
            # Dynamic dropdown for Plotly plot selection
            plotly_plot_options = ["Scatter Plot", "Line Plot", "Bar Plot"]
            selected_plotly_plot = st.selectbox("Select Plotly Plot", plotly_plot_options)
            # Plotly code based on user's selection
            if selected_plotly_plot == "Scatter Plot":
                st.write("### Scatter Plot")
                # Allow users to choose the x and y axes for the scatter plot
                scatterplot_x = st.selectbox("Select the X-axis column", data.columns)
                scatterplot_y = st.selectbox("Select the Y-axis column", data.columns)
                st.plotly_chart(px.scatter(data, x=data[scatterplot_x], y=data[scatterplot_y], title="Scatter Plot"))

            elif selected_plotly_plot == "Bar Plot":
                st.write("### Bar Plot")
                # Allow users to choose the x and y axes for the bar plot
                barplot_x = st.selectbox("Select the X-axis column", data.columns)
                barplot_y = st.selectbox("Select the Y-axis column", data.columns)
                st.plotly_chart(px.bar(data, x=data[barplot_x], y=data[barplot_y], title="Bar Plot"))

# Machine Learning tab
elif selected_tab == "Machine Learning":
    st.subheader("Machine Learning")
    st.sidebar.header("RESOURCE")
    uploaded_file = st.file_uploader("Upload autodata CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Machine learning
        st.subheader("Machine Learning")
        if st.checkbox("Build machine learning models"):
            target_variable = st.selectbox("Select the target variable", data.columns)
            features = data.drop(columns=[target_variable])

            # Model selection with dynamic options
            model_options = ["Random Forest", "Decision Tree", "Support Vector Machine"]
            selected_model = st.selectbox("Select Machine Learning Model", model_options)

            # Build selected model
            if selected_model == "Random Forest":
                model = RandomForestClassifier()
            # Add more models as needed

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
