# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Boston Housing dataset from CSV file
url = 'Boston_Housing.csv'
df = pd.read_csv(url)

# Create a function to preprocess the data
def preprocess_data(data):
    # Convert any non-numeric columns to numeric
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Scale the numerical features
    for col in data.columns[:-1]:
        data[col] = (data[col] - df[col].mean()) / df[col].std()

    return data
# Preprocess the data
# df = preprocess_data(df)

# Create a function to train and evaluate the model
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Create a Streamlit app
def app(medv=None):
    st.title('Boston Housing Price Prediction')
    st.sidebar.title('Menu')
    menu = st.sidebar.selectbox('Select an option', ('Data Exploration and Visualization', 'Prediction'))

    if menu == 'Data Exploration and Visualization':
        st.write('### Data Exploration and Visualization')

        # Add a checkbox for displaying the dataset
        if st.checkbox('Show Dataset'):
            st.write(df)

        # Add a histogram for the target variable (MEDV)
        st.write('### Histogram of Housing Prices')
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='medv', bins=50, kde=True, ax=ax)
        st.pyplot(fig)

        # Add a scatter plot for the relationship between RM and MEDV
        st.write('### Scatter Plot of RM vs. Housing Prices')
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='rm', y='medv', ax=ax)
        ax.set_xlabel('Average Number of Rooms per Dwelling')
        ax.set_ylabel('Median Value of Owner-Occupied Homes ($1000s)')
        st.pyplot(fig)

    elif menu == 'Prediction':
        st.write('### Prediction')

        # Add input fields for user to enter feature values
        crim = st.number_input('Per Capita Crime Rate:', min_value=0.0, max_value=100.0)
        zn = st.number_input('Proportion of Residential Land Zoned for Lots over 25,000 sq.ft.:', min_value=0.0, max_value=100.0)
        indus = st.number_input('Proportion of Non-Retail Business Acres per Town:', min_value=0.0, max_value=100.0)
        chas = st.selectbox('Charles River Dummy Variable (1 if Tract Bounds River; 0 Otherwise):', options=[0, 1])
        nox = st.number_input('Nitric Oxides Concentration (Parts per 10 Million):', min_value=0.0, max_value=100.0)
        rm = st.number_input('Average Number of Rooms per Dwelling:', min_value=0.0, max_value=20.0)
        age = st.number_input('Proportion of Owner-Occupied Units Built Prior to 1940:', min_value=0.0, max_value=100.0)
        dis = st.number_input('Weighted Distances to Five Boston Employment Centers:', min_value=0.0, max_value=100.0)
        rad = st.number_input('Index of Accessibility to Radial Highways:', min_value=0.0, max_value=100.0)
        tax = st.number_input('Full-Value Property Tax Rate per $10,000:', min_value=0.0, max_value=100.0)
        ptratio = st.number_input('Pupil-Teacher Ratio by Town:', min_value=0.0, max_value=100.0)
        black = st.number_input('Proportion of Blacks by Town:', min_value=0.0, max_value=100.0)
        lstat = st.number_input('% Lower Status of the Population:', min_value=0.0, max_value=100.0)

        # Create a DataFrame from the user input
        user_input = pd.DataFrame({
            'crim': crim,
            'zn': zn,
            'indus': indus,
            'chas': chas,
            'nox': nox,
            'rm': rm,
            'age': age,
            'dis': dis,
            'rad': rad,
            'tax': tax,
            'ptratio': ptratio,
            'black': black,
            'lstat': lstat,
        }, index=[0])

        # Split the data into training and testing sets
        X = df.drop('medv', axis=1)
        y = df['medv']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess the user input data
        user_input = preprocess_data(user_input)

        # Train and evaluate the model on the user input data
        model_type = st.selectbox('Select a Model Type:', options=['Linear Regression', 'Random Forest Regression'])
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Random Forest Regression':
            model = RandomForestRegressor(n_estimators=100)
        mse, r2 = train_model(model, X_train, X_test, y_train, y_test)

        # Make a prediction using the trained model and display the result
        prediction = model.predict(user_input)[0]
        st.write(f'Predicted Median Value of Owner-Occupied Homes: ${prediction:.2f}')
