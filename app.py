import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
# Function to load the dataset
@st.cache_data()
#ADDSOMESHIT
# def load_data():
#     url = 'higher+education+students+performance+evaluation/Student_dataset.csv'
#     return pd.read_csv(url)
# Function to describe the attribute information
#Mo ta dataset
def describe_attributes():
    st.write("## Đặc điểm Bộ Dữ Liệu")
    st.write("- Bộ dữ liệu bao gồm thông tin về các đặc điểm khác nhau của sinh viên Đại học Duy Tân, nhằm dự đoán điểm trung bình tích lũy (GPA) của họ.")
    st.write("- Bộ dữ liệu bao gồm các thuộc tính cá nhân như tuổi, giới tính, năm học, loại trường phổ thông tốt nghiệp, và tình trạng học bổng, cũng như hoạt động làm việc bán thời gian và tham gia hoạt động ngoại khóa.")
    st.write("- Biến mục tiêu là điểm GPA của sinh viên, được phân loại theo số.")
    st.write("- Bộ dữ liệu gồm 34 thuộc tính đầu vào.")
    st.write('===================================================================')
    st.write("## Thông Tin Thuộc Tính")
    st.write("1- Loại trường THPT mà bạn tốt nghiệp là gì?")
    st.write("2- Năm học qua Bạn có học bổng hay không?")
    st.write("3- Bạn có làm việc bán thời gian hay không?")
    st.write("4- Bạn có tham gia hoạt động nghệ thuật hoặc các môn thể thao đều đặn hay không?")
    st.write("5- Tổng thu nhập hiện tại của bạn là bao nhiêu (qua việc đi làm thêm hoặc đã có việc làm bán thời gian)?")
    st.write("6- Tình trạng hôn nhân của cha mẹ bạn là gì?")
    st.write("7- Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)")
    st.write("8- Tần suất đọc sách/báo không liên quan đến chuyên ngành mà bạn đang học")
    st.write("9- Tần suất đọc sách/báo liên quan đến chuyên ngành mà bạn đang học")
    st.write("10- Bạn có tham dự hội thảo/hội nghị liên quan đến ngành thường xuyên hay không")
    st.write("11- Ảnh hưởng của dự án/hoạt động cá nhân lên việc học của bạn")
    st.write("12- Bạn có đi học đều đặn không?")
    st.write("13- Để chuẩn bị cho kỳ thi thì bạn hay học với ai?")
    st.write("14- Thời điểm mà bạn ôn thi để chuẩn bị cho kỳ thi là?")
    st.write("15- Bạn có ghi chú khi học trên lớp không?")
    st.write("16- Bạn có tập trung nghe giảng trong lớp không?")
    st.write("17- Điểm trung bình tích lũy (GPA) trong học kỳ gần nhất của bạn là gì?")
    st.write("18- Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?")
    st.write("19- Trình độ tiếng Anh của bạn hiện tại (có thể ước đoán nếu chưa thi, chưa test)")
    st.write("20- Bạn học ngành gì? lĩnh vực gì?")

    st.write('===================================================================')

def explore_data(df):
    st.write("### Dataset Summary")
    st.write(df.head())
    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Dataset Description")
    st.write(df.describe())

    # Data Visualization
    st.write("### Data Visualization")
    
    # Histogram for Age Groups
    school_type_mapping = {
    0: 'Tư thục',
    1: 'Công lập',
    2: 'Trường chuyên'
    }
    df['Loại trường THPT mà bạn tốt nghiệp là gì?'] = df['Loại trường THPT mà bạn tốt nghiệp là gì?'].map(school_type_mapping)

    st.write("#### Biểu đồ cột thể hiện loại trường THPT")
    fig, ax = plt.subplots()
    highschool_type = df['Loại trường THPT mà bạn tốt nghiệp là gì?'].value_counts()
    highschool_type.plot(kind='bar', ax=ax)
    ax.set_xlabel('Loại trường')
    ax.set_ylabel('Tần suất')
    ax.set_xticklabels(highschool_type.index, rotation=0)
    st.pyplot(fig)

    #Hour Distribution
    study_hour_mapping = {
        0: '<5 giờ',
        1: '6-10 giờ',
        2: '11-20 giờ',
        3: 'hơn 20 giờ'
    }
    df['Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)'] = df['Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)'].map(study_hour_mapping)
    st.write("#### Biểu đồ cột thể hiện trung bình số giờ học mỗi tuần")  # Consider changing this title to "Bar Chart of Gender Distribution"
    fig, ax = plt.subplots()
    studyhour_counts = df['Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)'].value_counts()
    studyhour_counts.plot(kind='bar', ax=ax)  # Correct use of a bar chart for categorical data
    ax.set_xlabel('Số giờ')
    ax.set_ylabel('Tần suất')
    ax.set_xticklabels(studyhour_counts.index, rotation=0)
    st.pyplot(fig)

    expected_GPA_mapping = {
        0: '2.00-2.49',
        1: '2.50-2.99',
        2: '3.00-3.49',
        3: 'trên 3.49'
    }
    df['Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?'] = df['Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?'].map(expected_GPA_mapping)
    st.write("#### Bản đồ nhiệt tương quan")
    data_crosstab = pd.crosstab(df['Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)'], df['Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?'])
    fig, ax = plt.subplots()
    sns.heatmap(data_crosstab, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_title('Tần suất của số giờ học và điểm tích lũy mong đợi')
    st.pyplot(fig)

# Function to train and evaluate the model Randomforest
def train_and_evaluate_models(df):

    # Assuming 'GRADE' is the target variable
    X = df.drop(['Điểm trung bình tích lũy (GPA) trong học kỳ gần nhất của bạn là gì?'], axis=1)
    y = df['Điểm trung bình tích lũy (GPA) trong học kỳ gần nhất của bạn là gì?']

    # Encode categorical variables (assuming all are categorical or have been handled appropriately)
    X = pd.get_dummies(X, drop_first=True)
    
    # Save the feature columns in session state for later use during prediction
    if 'feature_columns' not in st.session_state:
        st.session_state['feature_columns'] = X.columns.tolist()

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to train
    models_to_train = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    trained_models = {}
    st.session_state.models = {}

    # Train and evaluate models
    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        # Directly update session state with each trained model
        st.session_state.models[name] = model
        y_pred = model.predict(X_test)

        # Calculate metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        # Display metrics
        st.write(f"#### {name} Performance")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"Accuracy: {accuracy:.4f}")

        trained_models[name] = model

    return trained_models

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    st.title("Student Performance Prediction")
    uploaded_file = st.file_uploader("Upload the dataset")
    # Check if a file has been uploaded
    models_trained = []
    # if 'models' not in st.session_state:
    #     st.session_state['models'] = {}
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        describe_attributes()
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
            high_school_type = st.selectbox("Loại trường THPT mà bạn tốt nghiệp là gì?", options=[ 'Tư thục','Công lập', 'Trường chuyên'], key='high_school_type')

        with col2:
            scholarship_type = st.selectbox("Năm học qua Bạn có học bổng hay không?", options=['Không','25%', '50%', '75%', 'Toàn phần'], key='scholarship_type')

        with col3:
            part_time_job = st.selectbox("Bạn có làm việc bán thời gian hay không?", options=['Có', 'Không'], key='part_time_job')

        col4, col5, col6 = st.columns(3)

        with col4:
            artistic_sports_activity = st.selectbox("Bạn có tham gia hoạt động nghệ thuật hoặc các môn thể thao đều đặn hay không?", options=['Có', 'Không'], key='artistic_sports_activity')

        with col5:
            total_income = st.selectbox("Tổng thu nhập hiện tại của bạn là bao nhiêu?", options=['Không có','Trên 2 triệu','Dưới 2 triệu'], key='total_income')

        with col6:
            parental_status = st.selectbox("Tình trạng hôn nhân của cha mẹ bạn là gì?", options=['Đã mất - một trong hai hoặc cả hai','Đã ly hôn','Đã kết hôn'], key='parental_status')

        col7, col8, col9 = st.columns(3)

        with col7:
            weekly_study_hours = st.selectbox("Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)", options=['<5 giờ', '6-10 giờ', '11-20 giờ', 'hơn 20 giờ'], key='weekly_study_hours')

        with col8:
            non_academic_reading = st.selectbox("Tần suất đọc sách/báo không liên quan đến chuyên ngành mà bạn đang học", options=['Không', 'Thỉnh thoảng', 'Thường xuyên'], key='non_academic_reading')

        with col9:
            academic_reading = st.selectbox("Tần suất đọc sách/báo liên quan đến chuyên ngành mà bạn đang học", options=['Không', 'Thỉnh thoảng', 'Thường xuyên'], key='academic_reading')

        col10, col11, col12 = st.columns(3)

        with col10:
            seminar_attendance = st.selectbox("Bạn có tham dự hội thảo/hội nghị liên quan đến ngành thường xuyên hay không", options=['Không', 'Có'], key='seminar_attendance')

        with col11:
            personal_project_impact = st.selectbox("Ảnh hưởng của dự án/hoạt động cá nhân lên việc học của bạn", options=['Tiêu cực', 'Trung lập', 'Tích cực'], key='personal_project_impact')

        with col12:
            class_attendance = st.selectbox("Bạn có đi học đều đặn không?", options=['Thỉnh thoảng', 'Luôn luôn'], key='class_attendance')

        col13, col14, col15 = st.columns(3)

        with col13:
            exam_preparation_with = st.selectbox("Để chuẩn bị cho kỳ thi thì bạn hay học với ai?", options=['Không áp dụng', 'Một mình', 'Với bạn bè' ], key='exam_preparation_with')

        with col14:
            exam_timing_preparation = st.selectbox("Thời điểm mà bạn ôn thi để chuẩn bị cho kỳ thi là?", options=['Không bao giờ ôn tập', 'Ngày gần nhất với kỳ thi', 'Đều đặn suốt học kỳ', ], key='exam_timing_preparation')

        with col15:
            note_taking = st.selectbox("Bạn có ghi chú khi học trên lớp không?", options=['Không bao giờ', 'Thỉnh thoảng', 'Luôn luôn'], key='note_taking')
       
        col16, col17, col18 = st.columns(3)

        with col16:
            class_listening = st.selectbox("Bạn có tập trung nghe giảng trong lớp không?", options=['Không bao giờ', 'Thỉnh thoảng', 'Luôn luôn'], key='class_listening')

        with col17:
            expected_graduation_gpa = st.selectbox("Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?", options=['2.00-2.49', '2.50-2.99', '3.00-3.49', 'trên 3.49'], key='expected_graduation_gpa')

        with col18:
            english_proficiency = st.selectbox("Trình độ tiếng Anh của bạn hiện tại (có thể ước đoán nếu chưa thi, chưa test)", options=['Cơ bản', 'Trung cấp (4.0- 6.0 IELTS)', 'Cao cấp (Mức IELTS > 6.0)'], key='english_proficiency')

        col19, col20 = st.columns(2)

        with col19:
            major_field_of_study = st.selectbox("Bạn học ngành gì? lĩnh vực gì?", options=[
                'Hệ thống Thông tin Quản lý (CMU)',
                'Kế toán',
                'Khoa học máy tính',
                'Tài chính-Ngân hàng (PSU)',
                'Kỹ thuật phần mềm (CMU)',
                'Kế toán (PSU)',
                'Marketing',
                'Kỹ thuật phần mềm',
                'Quản trị Kinh doanh (PSU)',
                'Kỹ thuật Xây dựng',
                'An toàn Thông tin',
                'Tài chính-Ngân hàng',
                'Kiểm toán',
                'Quản trị Kinh doanh',
                'Kiến trúc',
                'An toàn Thông tin (CMU)',
                'Hệ thống Thông tin Quản lý',
                'Du lịch',
                'Kinh doanh Thương mại',
                'Quản trị Khách sạn',
                'Quản trị Kinh doanh (ADP)',
                'Khoa học máy tính (ADP)',
                'Quản trị Khách sạn (ADP)'
            ], key='major_field_of_study')


        input_data = np.array([[
            high_school_type,                 # 4
            scholarship_type,                 # 5
            part_time_job,                    # 6
            artistic_sports_activity,         # 7
            total_income,                     # 9
            parental_status,                  # 15
            weekly_study_hours,               # 18
            non_academic_reading,             # 19
            academic_reading,                 # 20
            seminar_attendance,               # 21
            personal_project_impact,          # 22
            class_attendance,                 # 23
            exam_preparation_with,            # 24
            exam_timing_preparation,          # 25
            note_taking,                      # 26
            class_listening,                  # 27
            expected_graduation_gpa,          # 30
            english_proficiency,              # 31
            major_field_of_study,             # 32
        ]])

        # # Assuming 'GRADE' is the target variable
        # X = df.drop('GRADE', axis=1)
        # y = df['GRADE']
        
        # # Preprocessing steps (if not already done)
        # # Encode categorical variables (assuming all are categorical or have been handled appropriately)
        # X = pd.get_dummies(X, drop_first=True)

    if st.button("Predict Performance"):
    # Collecting input data
        input_features = {
            'Loại trường THPT mà bạn tốt nghiệp là gì?': [high_school_type],
            'Năm học qua Bạn có học bổng hay không?': [scholarship_type],
            'Bạn có làm việc bán thời gian hay không?': [part_time_job],
            'Bạn có tham gia hoạt động nghệ thuật hoặc các môn thể thao đều đặn hay không?': [artistic_sports_activity],
            'Tổng thu nhập hiện tại của bạn là bao nhiêu?': [total_income],
            'Tình trạng hôn nhân của cha mẹ bạn là gì?': [parental_status],
            'Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)': [weekly_study_hours],
            'Tần suất đọc sách/báo không liên quan đến chuyên ngành mà bạn đang học': [non_academic_reading],
            'Tần suất đọc sách/báo liên quan đến chuyên ngành mà bạn đang học': [academic_reading],
            'Bạn có tham dự hội thảo/hội nghị liên quan đến ngành thường xuyên hay không': [seminar_attendance],
            'Ảnh hưởng của dự án/hoạt động cá nhân lên việc học của bạn': [personal_project_impact],
            'Bạn có đi học đều đặn không?': [class_attendance],
            'Để chuẩn bị cho kỳ thi thì bạn hay học với ai?': [exam_preparation_with],
            'Thời điểm mà bạn ôn thi để chuẩn bị cho kỳ thi là?': [exam_timing_preparation],
            'Bạn có ghi chú khi học trên lớp không?': [note_taking],
            'Bạn có tập trung nghe giảng trong lớp không?': [class_listening],
            'Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?': [expected_graduation_gpa],
            'Trình độ tiếng Anh của bạn hiện tại (có thể ước đoán nếu chưa thi, chưa test)': [english_proficiency],
            'Bạn học ngành gì? lĩnh vực gì?': [major_field_of_study]
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame.from_dict(input_features)

        # Encode the input using get_dummies
        input_encoded = pd.get_dummies(input_df)

        # Make sure all training columns are present in the input
        missing_cols = set(st.session_state.feature_columns) - set(input_encoded.columns)
        for c in missing_cols:
            input_encoded[c] = 0

        # Reorder columns to match the training order
        input_encoded = input_encoded[st.session_state.feature_columns]

        # Prediction using the loaded model
        for name, model in st.session_state.models.items():
            prediction = model.predict(input_encoded)
            st.write(f"### Predicted Student's Performance using {name}:", prediction[0])
    else: 
        st.write("Please upload a file to proceed.")


if __name__ == "__main__":
    main()
