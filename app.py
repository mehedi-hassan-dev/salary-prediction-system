#----------------Import Libraries-----------------

import streamlit as st
import pandas as pd
import joblib


#-----------------------Page Configuration--------------------

st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="centered"
)


#------------------------Load Trained Model---------------------

model = joblib.load("models/best_model.pkl")


#------------------------Sidebar----------------------

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
### Salary Prediction System

This application predicts employee salary using a Machine Learning model.

### Model
✅ Tuned Random Forest Regressor

### Features Used
- Age
- Gender
- Education Level
- Job Title
- Years of Experience

### Tech Stack
- Python
- Scikit-Learn
- Streamlit
""")


#------------------Main Title----------------

st.title("💼 Salary Prediction System")

st.caption(
    "Predict employee salary using a Machine Learning model trained on employee information."
)

st.divider()


#--------------------User Input Section-----------------

st.subheader("📝 Employee Information")

# First Row

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=25
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

# Second Row

col3, col4 = st.columns(2)

with col3:
    education = st.selectbox(
        "Education Level",
        ["Bachelor's", "Master's", "PhD"]
    )

with col4:
    experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=1
    )

# Job Title

job_title = st.selectbox(
    "Job Title",
    [
        "Content Marketing Manager",
        "Data Analyst",
        "Data Scientist",
        "Digital Marketing Manager",
        "Director of HR",
        "Director of Marketing",
        "Financial Analyst",
        "Financial Manager",
        "Front end Developer",
        "Full Stack Engineer",
        "Human Resources Coordinator",
        "Human Resources Manager",
        "Junior HR Coordinator",
        "Junior HR Generalist",
        "Junior Sales Associate",
        "Junior Sales Representative",
        "Junior Web Developer",
        "Marketing Analyst",
        "Marketing Coordinator",
        "Marketing Manager",
        "Operations Manager",
        "Others",
        "Product Manager",
        "Research Director",
        "Research Scientist",
        "Sales Associate",
        "Sales Director",
        "Sales Executive",
        "Sales Manager",
        "Sales Representative",
        "Senior Data Scientist",
        "Senior HR Generalist",
        "Senior Human Resources Manager",
        "Senior Product Marketing Manager",
        "Senior Project Engineer",
        "Senior Software Engineer",
        "Software Developer",
        "Software Engineer",
        "Software Engineer Manager",
        "Web Developer"
    ]
)


#------------------Prediction Button-------------------

st.divider()

if st.button(
    "🚀 Predict Salary",
    use_container_width=True
):

#--------------Education Encoding---------------

    education_mapping = {
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }

    education_encoded = education_mapping[education]

#------------------Create Base Input Dictionary----------------

    input_data = {
        "Age": age,
        "Education Level": education_encoded,
        "Years of Experience": experience,

        "Gender_Male": 0,
        "Gender_Other": 0,

        "Job Title_Content Marketing Manager": 0,
        "Job Title_Data Analyst": 0,
        "Job Title_Data Scientist": 0,
        "Job Title_Digital Marketing Manager": 0,
        "Job Title_Director of HR": 0,
        "Job Title_Director of Marketing": 0,
        "Job Title_Financial Analyst": 0,
        "Job Title_Financial Manager": 0,
        "Job Title_Front end Developer": 0,
        "Job Title_Full Stack Engineer": 0,
        "Job Title_Human Resources Coordinator": 0,
        "Job Title_Human Resources Manager": 0,
        "Job Title_Junior HR Coordinator": 0,
        "Job Title_Junior HR Generalist": 0,
        "Job Title_Junior Sales Associate": 0,
        "Job Title_Junior Sales Representative": 0,
        "Job Title_Junior Web Developer": 0,
        "Job Title_Marketing Analyst": 0,
        "Job Title_Marketing Coordinator": 0,
        "Job Title_Marketing Manager": 0,
        "Job Title_Operations Manager": 0,
        "Job Title_Others": 0,
        "Job Title_Product Manager": 0,
        "Job Title_Research Director": 0,
        "Job Title_Research Scientist": 0,
        "Job Title_Sales Associate": 0,
        "Job Title_Sales Director": 0,
        "Job Title_Sales Executive": 0,
        "Job Title_Sales Manager": 0,
        "Job Title_Sales Representative": 0,
        "Job Title_Senior Data Scientist": 0,
        "Job Title_Senior HR Generalist": 0,
        "Job Title_Senior Human Resources Manager": 0,
        "Job Title_Senior Product Marketing Manager": 0,
        "Job Title_Senior Project Engineer": 0,
        "Job Title_Senior Software Engineer": 0,
        "Job Title_Software Developer": 0,
        "Job Title_Software Engineer": 0,
        "Job Title_Software Engineer Manager": 0,
        "Job Title_Web Developer": 0
    }

#---------------------Gender Encoding-----------------
                 
    if gender == "Male":
        input_data["Gender_Male"] = 1

    elif gender == "Other":
        input_data["Gender_Other"] = 1

    # Female = Base Category

#----------------------Job Title Encoding----------------

    job_column = f"Job Title_{job_title}"

    if job_column in input_data:
        input_data[job_column] = 1

#------------------------Convert to DataFrame---------------------

    input_df = pd.DataFrame([input_data])

#-----------------------Predict Salary-----------------------

    prediction = model.predict(input_df)[0]

#-----------------------Display Result---------------------

    st.metric(
        label="💰 Predicted Salary",
        value=f"Tk {prediction:,.0f}"
    )

    st.info(
        "Prediction generated using the Tuned Random Forest model."
    )


#-----------------------Footer-----------------------------

st.divider()

st.markdown(
    """
    <div style='text-align:center'>
        Built with using Streamlit & Scikit-Learn
        \nDeveloped by Md. Mehedi Hassan
    </div>
    """,
    unsafe_allow_html=True
)


