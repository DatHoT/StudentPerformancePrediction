import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    st.write("- The dataset consists of 145 instances and 31 input features.")
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

    # Assuming 'GRADE' is the target variable
    X = df.drop(['GRADE','STUDENT ID'], axis=1)
    y = df['GRADE']
    st.write(X)
    # Preprocessing steps (if not already done)
    # Encode categorical variables (assuming all are categorical or have been handled appropriately)
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to train
    models_to_train = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42)  # Replace Decision Tree with SVM
    }

    trained_models = {}

    # Train and evaluate models
    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        st.session_state.models = {}
        # Directly update session state with each trained model
        st.session_state.models[name] = model
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Display metrics
        st.write(f"#### {name} Performance")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        trained_models[name] = model
    st.write("Training completed. Models stored in session_state['models'].")
    # st.session_state['models'] = models
    # st.write("Models stored in session_state['models']")
    return trained_models

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


# Function to predict house prices using LinearRegression

# def predict_price(model, input_data):
#     # Ensure input_data has the same number of features as the training dataset
#     if input_data.shape[1] != model.coef_.shape[0]:
#         raise ValueError("Number of features in input data does not match the model")

#     prediction = model.predict(input_data)
#     return prediction

# # Function to predict house prices using RandomForest
# def predict_priceR(modelR, input_data):
#     predictionR = modelR.predict(input_data)
#     return predictionR

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
    models_trained = []
    # if 'models' not in st.session_state:
    #     st.session_state['models'] = {}
    if uploaded_file is not None:
        st.write("Current working directory:", os.getcwd())
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        #describe_attributes()
        explore_data(df)
        # Button to train and evaluate models
        st.write("### Model Training and Evaluation")
        if st.button('Train and Evaluate Models'):
            if 'df' in st.session_state and st.session_state['df'] is not None:
                models_trained = train_and_evaluate_models(st.session_state['df'])
                st.session_state['models'] = models_trained  # Re-assign to ensure update
            else:
                st.write("Please upload a dataset first.")

        st.write("### Student Performance Prediction")
        st.write("Enter the following features to predict the student's performance:")

        col1, col2, col3 = st.columns(3)

        with col1:
            student_age = st.selectbox("Student Age", options=[1, 2, 3], format_func=lambda x: {1: "18-21", 2: "22-25", 3: "above 26"}[x], key='student_age')

        with col2:
            sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: {1: "female", 2: "male"}[x], key='sex')

        with col3:
            high_school_type = st.selectbox("Graduated high-school type", options=[1, 2, 3], format_func=lambda x: {1: "private", 2: "state", 3: "other"}[x], key='high_school_type')

        col4, col5, col6 = st.columns(3)

        with col4:
            scholarship_type = st.selectbox("Scholarship type", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "None", 2: "25%", 3: "50%", 4: "75%", 5: "Full"}[x], key='scholarship_type')

        with col5:
            additional_work = st.selectbox("Additional work", options=[1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x], key='additional_work')

        with col6:
            artistic_sports_activity = st.selectbox("Regular artistic or sports activity", options=[1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x], key='artistic_sports_activity')

            # Continuing with the next row of inputs
        col7, col8, col9 = st.columns(3)

        with col7:
            have_partner = st.selectbox("Do you have a partner", options=[1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x], key='have_partner')

        with col8:
            total_salary = st.selectbox("Total salary if available", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "USD 135-200", 2: "USD 201-270", 3: "USD 271-340", 4: "USD 341-410", 5: "above 410"}[x], key='total_salary')

        with col9:
            transportation = st.selectbox("Transportation to the university", options=[1, 2, 3, 4], format_func=lambda x: {1: "Bus", 2: "Private car/taxi", 3: "bicycle", 4: "Other"}[x], key='transportation')

        col10, col11, col12 = st.columns(3)

        with col10:
            accommodation_type = st.selectbox("Accommodation type in Cyprus", options=[1, 2, 3, 4], format_func=lambda x: {1: "rental", 2: "dormitory", 3: "with family", 4: "Other"}[x], key='accommodation_type')

        with col11:
            mothers_education = st.selectbox("Mothers’ education", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: "primary school", 2: "secondary school", 3: "high school", 4: "university", 5: "MSc.", 6: "Ph.D."}[x], key='mothers_education')

        with col12:
            fathers_education = st.selectbox("Fathers’ education", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: "primary school", 2: "secondary school", 3: "high school", 4: "university", 5: "MSc.", 6: "Ph.D."}[x], key='fathers_education')

        col13, col14, col15 = st.columns(3)

        with col13:
            siblings = st.selectbox("Number of sisters/brothers", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "1", 2: "2", 3: "3", 4: "4", 5: "5 or above"}[x], key='siblings')

        with col14:
            parental_status = st.selectbox("Parental status", options=[1, 2, 3], format_func=lambda x: {1: "married", 2: "divorced", 3: "died - one of them or both"}[x], key='parental_status')

        with col15:
            mother_occupation = st.selectbox("Mother occupation", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: "retired", 2: "housewife", 3: "government officer", 4: "private sector employee", 5: "self-employment", 6: "other"}[x], key='mother_occupation')

        col16, col17, col18 = st.columns(3)

        with col16:
            father_occupation = st.selectbox("Father occupation", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "retired", 2: "government officer", 3: "private sector employee", 4: "self-employment", 5: "other"}[x], key='father_occupation')

        with col17:
            weekly_study_hours = st.selectbox("Weekly study hours", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "None", 2: "<5 hours", 3: "6-10 hours", 4: "11-20 hours", 5: "more than 20 hours"}[x], key='weekly_study_hours')

        with col18:
            reading_frequency_non_scientific = st.selectbox("Reading frequency (non-scientific books/journals)", options=[1, 2, 3], format_func=lambda x: {1: "None", 2: "Sometimes", 3: "Often"}[x], key='reading_frequency_non_scientific')

        col19, col20, col21 = st.columns(3)

        with col19:
            reading_frequency_scientific = st.selectbox("Reading frequency (scientific books/journals)", options=[1, 2, 3], format_func=lambda x: {1: "None", 2: "Sometimes", 3: "Often"}[x], key='reading_frequency_scientific')

        with col20:
            seminars_conferences_attendance = st.selectbox("Attendance to the seminars/conferences related to the department", options=[1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x], key='seminars_conferences_attendance')

        with col21:
            projects_activities_impact = st.selectbox("Impact of your projects/activities on your success", options=[1, 2, 3], format_func=lambda x: {1: "positive", 2: "negative", 3: "neutral"}[x], key='projects_activities_impact')

        col22, col23, col24 = st.columns(3)

        with col22:
            class_attendance = st.selectbox("Attendance to classes", options=[1, 2, 3], format_func=lambda x: {1: "always", 2: "sometimes", 3: "never"}[x], key='class_attendance')

        with col23:
            preparation_midterm_exams1 = st.selectbox("Preparation to midterm exams 1", options=[1, 2, 3], format_func=lambda x: {1: "alone", 2: "with friends", 3: "not applicable"}[x], key='preparation_midterm_exams1')

        with col24:
            preparation_midterm_exams2 = st.selectbox("Preparation to midterm exams 2", options=[1, 2, 3], format_func=lambda x: {1: "closest date to the exam", 2: "regularly during the semester", 3: "never"}[x], key='preparation_midterm_exams2')

        col25, col26, col27 = st.columns(3)

        with col25:
            taking_notes = st.selectbox("Taking notes in classes", options=[1, 2, 3], format_func=lambda x: {1: "never", 2: "sometimes", 3: "always"}[x], key='taking_notes')

        with col26:
            listening_in_classes = st.selectbox("Listening in classes", options=[1, 2, 3], format_func=lambda x: {1: "never", 2: "sometimes", 3: "always"}[x], key='listening_in_classes')

        with col27:
            discussion_contribution = st.selectbox("Discussion improves my interest and success in the course", options=[1, 2, 3], format_func=lambda x: {1: "never", 2: "sometimes", 3: "always"}[x], key='discussion_contribution')

        col28, col29, col30 = st.columns(3)

        with col28:
            flip_classroom_effectiveness = st.selectbox("Flip-classroom", options=[1, 2, 3], format_func=lambda x: {1: "not useful", 2: "useful", 3: "not applicable"}[x], key='flip_classroom_effectiveness')

        with col29:
            last_semester_gpa = st.selectbox("Cumulative grade point average in the last semester (/4.00)", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "<2.00", 2: "2.00-2.49", 3: "2.50-2.99", 4: "3.00-3.49", 5: "above 3.49"}[x], key='last_semester_gpa')

        with col30:
            expected_graduation_gpa = st.selectbox("Expected Cumulative grade point average in the graduation (/4.00)", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "<2.00", 2: "2.00-2.49", 3: "2.50-2.99", 4: "3.00-3.49", 5: "above 3.49"}[x], key='expected_graduation_gpa')

        # Since course_id doesn't naturally fit into the pattern of three columns, you can place it separately or adjust the layout as needed.
        course_id = st.selectbox("Course ID", options=[1, 2, 3, 4, 5, 6, 7, 8, 9], key='course_id')
    

        input_data = np.array([[student_age, sex, high_school_type, scholarship_type, additional_work, artistic_sports_activity, have_partner, total_salary, transportation, accommodation_type, mothers_education, fathers_education, siblings, parental_status, mother_occupation, father_occupation, weekly_study_hours, reading_frequency_non_scientific, reading_frequency_scientific, seminars_conferences_attendance, projects_activities_impact, class_attendance, preparation_midterm_exams1, preparation_midterm_exams2, taking_notes, listening_in_classes, discussion_contribution, flip_classroom_effectiveness, last_semester_gpa, expected_graduation_gpa, course_id]])

        # # Assuming 'GRADE' is the target variable
        # X = df.drop('GRADE', axis=1)
        # y = df['GRADE']
        
        # # Preprocessing steps (if not already done)
        # # Encode categorical variables (assuming all are categorical or have been handled appropriately)
        # X = pd.get_dummies(X, drop_first=True)

        if st.button("Predict Price"):
            for name in st.session_state.models:
                prediction = st.session_state.models[name].predict(input_data)
                st.write("### Predicted House Price using "+name+" :", prediction)
    else: 
        st.write("Please upload a file to proceed.")

if __name__ == "__main__":
    main()
