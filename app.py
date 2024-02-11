import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Function to load the dataset
@st.cache_data()
def load_data():
    url = 'higher+education+students+performance+evaluation/Student_dataset.csv'
    return pd.read_csv(url)
# Function to describe the attribute information
def describe_attributes():
    st.write("## Data Set Characteristics")
    st.write("- The dataset contains information about various features of university students, aimed at predicting their end-of-term academic results.")
    st.write("- It includes personal, family, and academic attributes such as age, sex, high-school type, scholarship type, study hours, reading frequency, and more.")
    st.write("- The target variable is the students' grades, categorized into several classes ranging from 'Fail' to 'AA'.")
    st.write("- The dataset consists of 145 instances and 33 input features.")
    st.write('===================================================================')
    st.write("## Attribute Information")
    st.write("1- Student Age (1: 18-21, 2: 22-25, 3: above 26)")
    st.write("2- Sex (1: female, 2: male)")
    st.write("3- Graduated high-school type: (1: private, 2: state, 3: other)")
    st.write("4- Scholarship type: (1: None, 2: 25%, 3: 50%, 4: 75%, 5: Full)")
    st.write("5- Additional work: (1: Yes, 2: No)")
    st.write("6- Regular artistic or sports activity: (1: Yes, 2: No)")
    st.write("7- Do you have a partner: (1: Yes, 2: No)")
    st.write("8- Total salary if available (1: USD 135-200, 2: USD 201-270, 3: USD 271-340, 4: USD 341-410, 5: above 410)")
    st.write("9- Transportation to the university: (1: Bus, 2: Private car/taxi, 3: bicycle, 4: Other)")
    st.write("10- Accommodation type in Cyprus: (1: rental, 2: dormitory, 3: with family, 4: Other)")
    st.write("11- Mothers’ education: (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)")
    st.write("12- Fathers’ education: (1: primary school, 2: secondary school, 3: high school, 4: university, 5: MSc., 6: Ph.D.)")
    st.write("13- Number of sisters/brothers (if available): (1: 1, 2:, 2, 3: 3, 4: 4, 5: 5 or above)")
    st.write("14- Parental status: (1: married, 2: divorced, 3: died - one of them or both)")
    st.write("15- Mother occupation: (1: retired, 2: housewife, 3: government officer, 4: private sector employee, 5: self-employment, 6: other)")
    st.write("16- Father occupation: (1: retired, 2: government officer, 3: private sector employee, 4: self-employment, 5: other)")
    st.write("17- Weekly study hours: (1: None, 2: <5 hours, 3: 6-10 hours, 4: 11-20 hours, 5: more than 20 hours)")
    st.write("18- Reading frequency (non-scientific books/journals): (1: None, 2: Sometimes, 3: Often)")
    st.write("19- Reading frequency (scientific books/journals): (1: None, 2: Sometimes, 3: Often)")
    st.write("20- Attendance to the seminars/conferences related to the department: (1: Yes, 2: No)")
    st.write("21- Impact of your projects/activities on your success: (1: positive, 2: negative, 3: neutral)")
    st.write("22- Attendance to classes (1: always, 2: sometimes, 3: never)")
    st.write("23- Preparation to midterm exams 1: (1: alone, 2: with friends, 3: not applicable)")
    st.write("24- Preparation to midterm exams 2: (1: closest date to the exam, 2: regularly during the semester, 3: never)")
    st.write("25- Taking notes in classes: (1: never, 2: sometimes, 3: always)")
    st.write("26- Listening in classes: (1: never, 2: sometimes, 3: always)")
    st.write("27- Discussion improves my interest and success in the course: (1: never, 2: sometimes, 3: always)")
    st.write("28- Flip-classroom: (1: not useful, 2: useful, 3: not applicable)")
    st.write("29- Cumulative grade point average in the last semester (/4.00): (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)")
    st.write("30- Expected Cumulative grade point average in the graduation (/4.00): (1: <2.00, 2: 2.00-2.49, 3: 2.50-2.99, 4: 3.00-3.49, 5: above 3.49)")
    st.write("31- Course ID")
    st.write("32- OUTPUT Grade (0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA)")
    st.write('===================================================================')
# Function to explore the dataset
def explore_data(df):
    describe_attributes()
    st.write("### Dataset Summary")
    st.write(df.head())
    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Dataset Description")
    st.write(df.describe())

     # Data visualization
    st.write("### Data Visualization")
    st.write("#### Histogram for Age Groups")
    fig, ax = plt.subplots()
    # Assuming '1' is the column for student age groups
    counts, bins, patches = ax.hist(df['1'], bins=range(1, 5), rwidth=0.8, align='left')
    ax.set_xlabel('Age Groups')
    ax.set_ylabel('Frequency')
    # Set x-ticks to be at the center of each bin
    ax.set_xticks(np.arange(1, 4) + 0.5)
    ax.set_xticklabels(['18-21', '22-25', 'above 26'])
    st.pyplot(fig)

    st.write("#### Gender Distribution")
    fig, ax = plt.subplots()
    df['2'].value_counts().plot(kind='bar', ax=ax)  # Assuming '2' is the column for sex (1: female, 2: male)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(['Female', 'Male'], rotation=0)
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to train and evaluate the model
def train_model(df):
    st.write("### Model Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("#### Model Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model, "LinearRegression.pkl")
    return model

# Function to train and evaluate the model Randomforest
def train_modelR(df):
    st.write("### Model Randomforest Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelR = RandomForestRegressor(n_estimators=100, random_state=42)
    modelR.fit(X_train, y_train)

    y_pred = modelR.predict(X_test)

    st.write("#### Model Randomforest Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(modelR, "RandomForest.pkl")
    return modelR

# Function to predict house prices using LinearRegression

def predict_price(model, input_data):
    # Ensure input_data has the same number of features as the training dataset
    if input_data.shape[1] != model.coef_.shape[0]:
        raise ValueError("Number of features in input data does not match the model")

    prediction = model.predict(input_data)
    return prediction

# Function to predict house prices using RandomForest
def predict_priceR(modelR, input_data):
    predictionR = modelR.predict(input_data)
    return predictionR

# Function to visualize the predicted prices
def visualize_prediction(df, predicted_prices):
    sorted_indices = np.argsort(df['RM'])
    sorted_predicted_prices = predicted_prices.flatten()[sorted_indices]

    fig, ax = plt.subplots()
    ax.scatter(df['RM'], df['PRICE'], label='Actual')
    ax.scatter(df['RM'].iloc[sorted_indices], sorted_predicted_prices, color='red', label='Predicted')
    ax.set_xlabel('RM')
    ax.set_ylabel('PRICE')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("House Price Prediction")
    df = load_data()
    #describe_attributes()
    explore_data(df)
    model = train_model(df)
    modelR = train_modelR(df)

    st.write("### House Price Prediction")
    st.write("Enter the following features to get the predicted price:")
    crim = st.number_input("CRIM - Per Capita Crime Rate:", value=0.0, step=0.01)
    zn = st.number_input("ZN - Proportion of Residential Land Zoned:", value=0.0, step=0.5)
    indus = st.number_input("INDUS - Proportion of Non-Retail Business Acres:", value=0.0, step=0.01)
    chas = st.selectbox("CHAS - Charles River Dummy Variable:", options=[0, 1])
    nox = st.number_input("NOX - Nitric Oxides Concentration (parts per 10 million):", value=0.0, step=0.01)
    rm = st.number_input("RM - Average Number of Rooms per Dwelling:", value=0.0, step=0.01)
    age = st.number_input("AGE - Proportion of Owner-Occupied Units Built Prior to 1940:", value=0.0, step=0.01)
    dis = st.number_input("DIS - Weighted Distances to Five Boston Employment Centers:", value=0.0, step=0.01)
    rad = st.number_input("RAD - Index of Accessibility to Radial Highways:", value=0.0, step=1.0)
    tax = st.number_input("TAX - Full-Value Property Tax Rate per $10,000:", value=0.0, step=1.0)
    ptratio = st.number_input("PTRATIO - Pupil-Teacher Ratio by Town:", value=0.0, step=0.01)
    b = st.number_input("B - Proportion of Blacks:", value=0.0, step=0.01)
    lstat = st.number_input("LSTAT - Percentage of Lower Status of the Population:", value=0.0, step=0.01)
    medv = st.number_input("Median value of owner-occupied homes in $1000's:", value=0.0, step=0.01)

    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv]])

    if st.button("Predict Price"):
        prediction = predict_price(model, input_data)
        st.write("### Predicted House Price using LinearRegression:", prediction)

        prediction = predict_priceR(modelR, input_data)
        st.write("### Predicted House Price using RandomForest:", prediction)

if __name__ == "__main__":
    main()
