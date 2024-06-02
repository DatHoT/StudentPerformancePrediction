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
    st.write("- Bộ dữ liệu gồm 225 mục nhập và 34 thuộc tính đầu vào.")
    st.write('===================================================================')
    st.write("## Thông Tin Thuộc Tính")
    st.write("1- Thời gian hoàn thành khảo sát")
    st.write("2- Tuổi của sinh viên")
    st.write("3- Giới tính")
    st.write("4- Năm học hiện tại")
    st.write("5- Loại trường phổ thông đã tốt nghiệp")
    st.write("6- Tình trạng học bổng năm ngoái")
    st.write("7- Tình trạng làm việc bán thời gian")
    st.write("8- Tham gia hoạt động nghệ thuật hoặc thể thao")
    st.write("9- Tình trạng hôn nhân")
    st.write("10- Tổng thu nhập hiện tại (từ việc làm thêm hoặc đã có việc làm)")
    st.write("11- Phương tiện đi lại đến trường")
    st.write("12- Loại chỗ ở hiện tại")
    st.write("13- Trình độ học vấn của mẹ")
    st.write("14- Trình độ học vấn của cha")
    st.write("15- Số lượng anh chị em")
    st.write("16- Tình trạng hôn nhân của cha mẹ")
    st.write("17- Nghề nghiệp của mẹ")
    st.write("18- Nghề nghiệp của cha")
    st.write("19- Số giờ học trung bình mỗi tuần")
    st.write("20- Tần suất đọc sách/báo không liên quan đến chuyên ngành")
    st.write("21- Tần suất đọc sách/báo liên quan đến chuyên ngành")
    st.write("22- Tham gia thường xuyên các hội thảo/hội nghị liên quan đến ngành")
    st.write("23- Ảnh hưởng của dự án/hoạt động cá nhân đến việc học")
    st.write("24- Điều đặn đi học")
    st.write("25- Người bạn học chung khi chuẩn bị cho kỳ thi")
    st.write("26- Thời gian ôn tập chuẩn bị cho kỳ thi")
    st.write("27- Ghi chú trong lớp")
    st.write("28- Tập trung nghe giảng trong lớp")
    st.write("29- Quan điểm về 'Tham gia thảo luận giúp tôi tập trung và thành công hơn trong khóa học'")
    st.write("30- Đánh giá về phương pháp lớp học đảo ngược (học tại nhà, thảo luận tại lớp)")
    st.write("31- GPA gần nhất")
    st.write("32- GPA mong đợi khi tốt nghiệp")
    st.write("33- Trình độ tiếng Anh hiện tại")
    st.write("34- Ngành và lĩnh vực học tập")
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
    st.write("#### Biểu đồ cột thể hiện Nhóm độ tuổi")
    fig, ax = plt.subplots()
    age_group_counts = df['Tuổi của bạn'].value_counts()
    age_group_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Nhóm độ tuổi')
    ax.set_ylabel('Tần suất')
    ax.set_xticklabels(age_group_counts.index, rotation=0)
    st.pyplot(fig)

    # Gender Distribution
    st.write("#### Biểu đồ cột phân bố giới tính")  # Consider changing this title to "Bar Chart of Gender Distribution"
    fig, ax = plt.subplots()
    gender_counts = df['Giới tính'].value_counts()
    gender_counts.plot(kind='bar', ax=ax)  # Correct use of a bar chart for categorical data
    ax.set_xlabel('Giới tính')
    ax.set_ylabel('Tần suất')
    ax.set_xticklabels(gender_counts.index, rotation=0)
    st.pyplot(fig)

    # Exclude the 'Timestamp' column before calculating correlations
    numeric_df = df.drop(columns=['Timestamp']).select_dtypes(include=[np.number])

    # Heatmap for Correlations
    st.write("#### Bản đồ nhiệt tương quan")
    data_crosstab = pd.crosstab(df['Giới tính'], df['Bạn đã lập gia đình chưa?'])
    fig, ax = plt.subplots()
    sns.heatmap(data_crosstab, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_title('Tần suất của giới tình và tình trạng hôn nhân')
    st.pyplot(fig)


    # Scatter Plot for GPA comparison
    st.write("#### Scatter Plot so sánh GPA hiện tại và mong đợi")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Điểm trung bình tích lũy (GPA) trong học kỳ gần nhất của bạn là gì?', y='Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?', ax=ax)
    ax.set_xlabel('GPA hiện tại')  # Set the label for the x-axis
    ax.set_ylabel('GPA mong đợi lúc tốt nghiệp')  # Set the label for the y-axis
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
        #describe_attributes()
        #explore_data(df)
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
            student_age = st.selectbox("Tuổi của bạn", options=['18-21', '22-25', 'trên 26'], key='student_age')

        with col2:
            sex = st.selectbox("Giới tính", options=['Nữ', 'Nam', 'Khác'], key='sex')

        with col3:
            year_in_school = st.selectbox("Bạn hiện đang là sinh viên năm thứ mấy?", options=[1, 2, 3, 4], key='year_in_school')

        col4, col5, col6 = st.columns(3)

        with col4:
            high_school_type = st.selectbox("Loại trường THPT mà bạn tốt nghiệp là gì?", options=['Công lập', 'Tư thục', 'Trường chuyên'], key='high_school_type')

        with col5:
            scholarship_type = st.selectbox("Năm học qua Bạn có học bổng hay không?", options=['Không', 'Toàn phần', '25%', '50%', '75%'], key='scholarship_type')

        with col6:
            part_time_job = st.selectbox("Bạn có làm việc bán thời gian hay không?", options=['Có', 'Không'], key='part_time_job')

        col7, col8, col9 = st.columns(3)

        with col7:
            artistic_sports_activity = st.selectbox("Bạn có tham gia hoạt động nghệ thuật hoặc các môn thể thao đều đặn hay không?", options=['Có', 'Không'], key='artistic_sports_activity')

        with col8:
            marital_status = st.selectbox("Bạn đã lập gia đình chưa?", options=['Chưa', 'Có'], key='marital_status')

        with col9:
            total_income = st.selectbox("Tổng thu nhập hiện tại của bạn là bao nhiêu?", options=['Dưới 2 triệu', 'Không có', 'Trên 2 triệu'], key='total_income')

        col10, col11, col12 = st.columns(3)

        with col10:
            transportation = st.selectbox("Phương tiện di chuyển đến trường là gì?", options=['Xe máy', 'Xe đạp', 'Xe ô tô', 'Khác'], key='transportation')

        with col11:
            accommodation_type = st.selectbox("Loại chỗ hiện tại của bạn là gì?", options=['Thuê nhà', 'Với gia đình', 'Khác', 'Ký túc xá'], key='accommodation_type')

        with col12:
            mother_education_level = st.selectbox("Trình độ học vấn của mẹ bạn là gì?", options=['Tiểu học', 'Trung học cơ sở', 'Trung học phổ thông', 'Đại học', 'Thạc sĩ', 'Tiến sĩ', 'Khác'], key='mother_education_level')

        col13, col14, col15 = st.columns(3)

        with col13:
            father_education_level = st.selectbox("Trình độ Học vấn của Ba bạn là gì?", options=['Tiểu học', 'Trung học cơ sở', 'Trung học phổ thông', 'Đại học', 'Thạc sĩ', 'Tiến sĩ', 'Khác'], key='father_education_level')

        with col14:
            siblings = st.selectbox("Số anh chị em của Bạn", options=['1', '2', '3', '4', 'Không có', '5 trở lên'], key='siblings')

        with col15:
            parental_status = st.selectbox("Tình trạng hôn nhân của cha mẹ bạn là gì?", options=['Đã kết hôn', 'Đã ly hôn', 'Đã mất - một trong hai hoặc cả hai'], key='parental_status')

        col16, col17, col18 = st.columns(3)

        with col16:
            mother_occupation = st.selectbox("Nghề nghiệp của mẹ", options=['Khác', 'Công chức, nhân viên', 'Tự kinh doanh', 'Đã nghỉ hưu', 'Nội trợ', 'Lãnh đạo'], key='mother_occupation')

        with col17:
            father_occupation = st.selectbox("Nghề nghiệp của cha", options=['Khác', 'Công chức', 'Tự kinh doanh', 'Quản lý', 'Đã nghỉ hưu', 'Lãnh đạo'], key='father_occupation')

        with col18:
            weekly_study_hours = st.selectbox("Trung bình số giờ học mỗi tuần của bạn (giờ học trên lớp)", options=['<5 giờ', '6-10 giờ', '11-20 giờ', 'hơn 20 giờ'], key='weekly_study_hours')

        col19, col20, col21 = st.columns(3)

        with col19:
            non_academic_reading = st.selectbox("Tần suất đọc sách/báo không liên quan đến chuyên ngành mà bạn đang học", options=['Không', 'Thỉnh thoảng', 'Thường xuyên'], key='non_academic_reading')

        with col20:
            academic_reading = st.selectbox("Tần suất đọc sách/báo liên quan đến chuyên ngành mà bạn đang học", options=['Không', 'Thỉnh thoảng', 'Thường xuyên'], key='academic_reading')

        with col21:
            seminar_attendance = st.selectbox("Bạn có tham dự hội thảo/hội nghị liên quan đến ngành thường xuyên hay không", options=['Không', 'Có'], key='seminar_attendance')

        col22, col23, col24 = st.columns(3)

        with col22:
            personal_project_impact = st.selectbox("Ảnh hưởng của dự án/hoạt động cá nhân lên việc học của bạn", options=['Tiêu cực', 'Trung lập', 'Tích cực'], key='personal_project_impact')

        with col23:
            class_attendance = st.selectbox("Bạn có đi học đều đặn không?", options=['Luôn luôn', 'Thỉnh thoảng'], key='class_attendance')

        with col24:
            exam_preparation_with = st.selectbox("Để chuẩn bị cho kỳ thi thì bạn hay học với ai?", options=['Một mình', 'Với bạn bè', 'Không áp dụng'], key='exam_preparation_with')

        col25, col26, col27 = st.columns(3)

        with col25:
            exam_timing_preparation = st.selectbox("Thời điểm mà bạn ôn thi để chuẩn bị cho kỳ thi là?", options=['Ngày gần nhất với kỳ thi', 'Đều đặn suốt học kỳ', 'Không bao giờ ôn tập'], key='exam_timing_preparation')

        with col26:
            note_taking = st.selectbox("Bạn có ghi chú khi học trên lớp không?", options=['Không bao giờ', 'Thỉnh thoảng', 'Luôn luôn'], key='note_taking')

        with col27:
            class_listening = st.selectbox("Bạn có tập trung nghe giảng trong lớp không?", options=['Không bao giờ', 'Thỉnh thoảng', 'Luôn luôn'], key='class_listening')

        col28, col29, col30 = st.columns(3)

        with col28:
            class_discussion_impact = st.selectbox("Bạn nghĩ thể nào về quan điểm 'Phát biểu thảo luận giúp tôi quan tâm và thành công hơn trong khóa học'?", options=['Không đồng ý', 'Đồng ý một phần', 'Đồng ý'], key='class_discussion_impact')

        with col29:
            flipped_classroom_effectiveness = st.selectbox("Bạn thấy phương pháp flipped classroom có ích hay không?", options=['Không hữu ích', 'Không áp dụng', 'Hữu ích'], key='flipped_classroom_effectiveness')

        with col30:
            expected_graduation_gpa = st.selectbox("Điểm trung bình tích lũy mong đợi khi tốt nghiệp của bạn là gì?", options=['<2.00', '2.00-2.49', '2.50-2.99', '3.00-3.49', 'trên 3.49'], key='expected_graduation_gpa')

        col31, col32, col33 = st.columns(3)

        with col31:
            english_proficiency = st.selectbox("Trình độ tiếng Anh của bạn hiện tại (có thể ước đoán nếu chưa thi, chưa test)", options=['Cơ bản', 'Trung cấp (4,0- 6.0 IELTS)', 'Cao cấp (Mức IELTS > 6.0)'], key='english_proficiency')

        # Since course_id doesn't naturally fit into the pattern of three columns, you can place it separately or adjust the layout as needed.
        with col32:
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
            student_age,                      # 1
            sex,                              # 2
            year_in_school,                   # 3
            high_school_type,                 # 4
            scholarship_type,                 # 5
            part_time_job,                    # 6
            artistic_sports_activity,         # 7
            marital_status,                   # 8
            total_income,                     # 9
            transportation,                   # 10
            accommodation_type,               # 11
            mother_education_level,           # 12
            father_education_level,           # 13
            siblings,                         # 14
            parental_status,                  # 15
            mother_occupation,                # 16
            father_occupation,                # 17
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
            class_discussion_impact,          # 28
            flipped_classroom_effectiveness,  # 29
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
            # Recreate dummy variables to match training
            input_encoded = pd.get_dummies(input)
            
            # Ensure all training columns are present in the input
            missing_cols = set(st.session_state.feature_columns) - set(input_encoded.columns)
            for c in missing_cols:
                input_encoded[c] = 0
            
            # Reorder columns to match the training order
            input_encoded = input_encoded[st.session_state.feature_columns]
            
            # Prediction
            for name in st.session_state.models:
                prediction = st.session_state.models[name].predict(input_encoded)
                st.write(f"### Predicted Student's Performance using {name}:", prediction)
    else: 
        st.write("Please upload a file to proceed.")

if __name__ == "__main__":
    main()
