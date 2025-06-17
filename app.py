import pandas as pd
import numpy as np
import os
import pickle
import streamlit as st

from streamlit_option_menu import option_menu


# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# loading the saved models

kidney_disease_model = pickle.load(open(f'D:\python\pro4\kidney_disease_LR_model.pkl', 'rb'))

liver_disease_model = pickle.load(open(f'D:\python\pro4\liver_disease_LR_model.pkl', 'rb'))

parkinsons_model = pickle.load(open(f'D:\python\pro4\parkinsons_disease_LR_model.pkl', 'rb'))

import streamlit as st

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Kidney Disease", "Liver Disease", "Parkinson's Disease"],
        icons=["house", "heart-pulse", "droplet-half", "person-wheelchair"],  # Bootstrap icon names
        menu_icon="hospital",  # Icon for the menu title
        default_index=0
    )

st.write(f"You selected: {selected}")

if selected == 'Home':
    st.title("🩺 Multiple Disease Prediction System")
    st.subheader("Welcome to the Health Risk Assessment Tool")

    st.markdown("""
    This Streamlit web application allows users to **predict the likelihood of three diseases** using Machine Learning models:

    - 🧪 **Kidney Disease**
    - 🧬 **Liver Disease**
    - 🧠 **Parkinson’s Disease**

    ### 🔍 How It Works
    - Go to the respective disease page using the **sidebar menu or slider**
    - Fill in the required medical parameters
    - Get instant predictions powered by trained machine learning models

    ### 🛡️ Disclaimer
    - This app is for **educational and informational purposes only**
    - It does **not replace professional medical advice, diagnosis, or treatment**

    ### 📢 Tip
    - Ensure all inputs are accurate for the best prediction results.
    """)

    st.info("Navigate using the sidebar to try out each prediction model.")

# Kidney Disease Page
if selected == 'Kidney Disease':
    # page title
    st.title('🫘 Kidney Disease Prediction')
    st.write("Enter the required details to predict kidney disease.")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120)
    bp = st.number_input("Blood Pressure", min_value=40, max_value=200)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    hemo = st.number_input("Hemoglobin", min_value=3.0, max_value=20.0)
    pvc = st.selectbox("Pus Cell Clumps (pvc)", ['present', 'notpresent'])
    pot = st.number_input("Potassium (pot)", min_value=2.0, max_value=10.0)

    # Encoding 'pvc'
    pvc_encoded = 1 if pvc == 'present' else 0

    # When Predict button is clicked
    if st.button("Predict"):
        input_data = np.array([[age, bp, sg, al, su, hemo, pvc_encoded, pot]])
        prediction = kidney_disease_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Prediction: Kidney Disease Detected")
        else:
            st.success("Prediction: No Kidney Disease")


# Liver Disease Page
if selected  == 'Liver Disease':
    st.header("🤒 Liver Disease Prediction")
    # Add your model input fields and prediction logic here
    st.write("Enter the required details to predict liver disease.")


    # Input fields with unique keys
    age = st.number_input("Age", min_value=1, max_value=120, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    total_bilirubin = st.number_input("Total Bilirubin", key="total_bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin", key="direct_bilirubin")
    alk_phosphotase = st.number_input("Alkaline Phosphotase", key="alk_phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", key="alt")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", key="ast")
    total_proteins = st.number_input("Total Proteins", key="total_proteins")
    albumin = st.number_input("Albumin", key="albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio", key="ag_ratio")

    # Preprocess gender
    gender_numeric = 1 if gender == "Male" else 0

    # Predict button
    if st.button("Predict", key="predict_liver"):
        input_data = np.array([[age, gender_numeric, total_bilirubin, direct_bilirubin,
                                alk_phosphotase, alamine_aminotransferase, aspartate_aminotransferase,
                                total_proteins, albumin, ag_ratio]])

        prediction = liver_disease_model.predict(input_data)[0]

        if prediction == 1:
            st.error("The person is likely to have liver disease.")
        else:
            st.success("The person is not likely to have liver disease.")

# Parkinson's Disease Page
if selected == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    # Add your model input fields and prediction logic here
    st.write("Enter the required details to predict Parkinson's disease.")

    def user_input_features():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fo = st.slider('MDVP:Fo(Hz)', 80, 250, 120)
        with col2:
            fhi = st.slider('MDVP:Fhi(Hz)', 100, 300, 150)
        with col3:
            flo = st.slider('MDVP:Flo(Hz)', 60, 200, 100)
        with col4:
            jitter_percent = st.slider('MDVP:Jitter(%)', 0.0, 0.02, 0.01)
        with col1:
            shimmer = st.slider('MDVP:Shimmer', 0.0, 0.1, 0.05)
        with col2:
            hnr = st.slider('HNR', 0.0, 40.0, 20.0)
        with col3:
            rpde = st.slider('RPDE', 0.0, 1.0, 0.5)
        with col4:
            dfa = st.slider('DFA', 0.0, 1.0, 0.6)
        with col1:
            spread1 = st.slider('spread1', -7.0, -1.0, -4.0)
        with col2:
            spread2 = st.slider('spread2', 0.0, 1.0, 0.3)
        with col3:
            d2 = st.slider('D2', 1.0, 3.0, 2.0)
        with col4:
            PPE = st.slider('PPE', 0.0, 0.6, 0.2)

        data = {
            'MDVP:Fo(Hz)': fo,
            'MDVP:Fhi(Hz)': fhi,
            'MDVP:Flo(Hz)': flo,
            'MDVP:Jitter(%)': jitter_percent,
            'MDVP:Shimmer': shimmer,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': d2,
            'PPE': PPE
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display input data
    st.subheader('User Input Parameters')
    st.write(input_df)

    # Make prediction
    if st.button('Predict', key='predict_button'):
        prediction = parkinsons_model.predict(input_df)
        result = 'Parkinson’s Disease Detected' if prediction[0] == 1 else 'No Parkinson’s Disease'
        st.subheader('Prediction Result')
        st.success(result)
