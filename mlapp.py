import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Read the Boston Housing dataset from Github
# url = "D:\\Data\\MLProject\\Boston_Housing.csv"
df = pd.read_csv("Boston_Housing.csv")

# Create a dropdown menu for selecting the input features
feature_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
selected_features = st.multiselect('Select Input Features:', feature_cols)

# Display the selected features
st.write('You selected:', selected_features)

# Create a slider for selecting the target variable
target_col = 'MEDV'
target = st.slider('Select Target Variable:', float(df[target_col].min()), float(df[target_col].max()), float(df[target_col].mean()))

# Train a linear regression model on the selected features and target variable
X = df[selected_features]
y = df[target_col]
model = LinearRegression()
model.fit(X, y)

# Create input fields for the selected features
input_dict = {}
for col in selected_features:
    input_dict[col] = st.number_input(col, value=0.0)

# Make a prediction based on the user input
input_df = pd.DataFrame(input_dict, index=[0])
prediction = model.predict(input_df)

# Display the predicted price
st.write('Predicted Price:', prediction[0])
