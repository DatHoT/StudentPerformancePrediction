import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Function to load the dataset
@st.cache_data()
# def load_data():
#     url = 'higher+education+students+performance+evaluation/Student_dataset.csv'
#     return pd.read_csv(url)
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

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to train and evaluate the model Randomforest
def train_and_evaluate_models(df):
    st.write("### Model Training and Evaluation")

    # Assuming 'GRADE' is the target variable
    X = df.drop('GRADE', axis=1)
    y = df['GRADE']
    
    # Preprocessing steps (if not already done)
    # Encode categorical variables (assuming all are categorical or have been handled appropriately)
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    trained_models = {}

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Display metrics
        st.write(f"#### {name} Performance")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        trained_models[name] = model

        # Save model
    st.session_state['models'] = trained_models

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


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
    st.title("Student Performance Prediction")
    uploaded_file = st.file_uploader("Upload the dataset")
    # Check if a file has been uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        #describe_attributes()
        explore_data(df)
        # Initialize or load session state for models
        if 'models' not in st.session_state:
            st.session_state.models = {}

        # Button to train and evaluate models
        if st.button('Train and Evaluate Models'):
            if 'df' in st.session_state:  # Check if df is loaded
                st.session_state.models = train_and_evaluate_models(st.session_state['df'])
                st.write("Models trained and evaluated.")
            else:
                st.write("Please upload a dataset first.")

        # Assuming you want to save all models, iterate through the models in session state
        if st.button('Save Models'):
            if 'models' in st.session_state and st.session_state.models:  # Check if models exist in session state
                for name, model in st.session_state.models.items():
                    filename = f"{name.replace(' ', '_')}.pkl"
                    save_model(model, filename)
                    st.write(f"{name} model saved.")
            else:
                st.write("No models to save. Train models first.")

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
    else: 
        st.write("Please upload a file to proceed.")

if __name__ == "__main__":
    main()
