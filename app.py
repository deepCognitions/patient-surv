
import streamlit as st
import shap
import pickle
import tensorflow
import pandas as pd
import numpy as np
import base64
import joblib
import streamlit.components.v1 as components
from predict import get_prediction, explain_model_prediction
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model
nn = load_model('Model/DL_PSP.h5')

open_file = open("Data/featuresf", "rb")
features = pickle.load(open_file)
open_file.close()

#shap.initjs()      

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



st.set_page_config(page_title="Deep =>  Patient Survival Prediction",
                   page_icon="🚑", layout="wide" )
                   
def main():
    
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            background-color: #AEB6BF;
          
        }
        
        .logo-text {
            font-weight:600 !important;
            font-size:45px !important;
            color: #ECF0F1 !important;
            padding-top: 150px !important;
            padding-left: 75px !important;
            
        }
        .sub-text {
            font-weight:150 !important;
            font-size:15px !important;
            color: #000000 !important;
            padding-bottom: 20px !important;
            padding-right: 20px !important;
            
        }
        .logo-img {
            float:right;
            width:50%;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('Img/bg2.png', "rb").read()).decode()}">
            <p class="logo-text">Site's CO2 Emission Evaluation</p>
            
        </div>
        """,
        unsafe_allow_html=True
    )
        
    with st.form('prediction_form'):
    
        st.header("Enter the input for following info:")

        col1, col2, col3= st.columns(3)
        
        with col1:
      
            inp = {}
            inp['d1_lactate_max'] = st.number_input("Highest Lactate Concentration :", value = 2.50, format="%.2f")
            inp['d1_lactate_min'] = st.number_input("Lowest Lactate Concentration :", value = 2.50, format="%.2f")
            inp['apache_4a_hospital_death_prob'] = st.number_input("Hospital Death Prob (Apache 4a) :", value = 0.00, format="%.2f")
            inp['gcs_motor_apache'] = st.number_input("GCS Motor Component :", value = 5.00, format="%.2f")
            inp['gcs_eyes_apache'] = st.number_input("GCS Eyes Component :", value = 2.50, format="%.2f")
            
        with col2:    
            inp['apache_4a_icu_death_prob'] = st.number_input("ICU Death Prob (Apache 4a) :", value = -1.00, format="%.2f")
            inp['gcs_verbal_apache'] = st.number_input("GCS Verbal Component :", value = 3.00, format="%.2f")
            inp['ventilated_apache'] = st.selectbox("Ventilated invasively :", ('No', 'Yes'))
            inp['d1_spo2_min'] = st.number_input("Lowest peripheral O2 concentration :", value = 83.00, format="%.2f", step=1.00)
            inp['d1_sysbp_min'] = st.number_input("Lowest Diastolic BP :", value = 97.00, format="%.2f", step=1.00)
            
        with col3:    
            inp['d1_sysbp_noninvasive_min'] = st.number_input("Lowest Diastolic BP(non-invasively measured) :", value = 97.00, format="%.2f", step=1.00)
            inp['d1_temp_min'] = st.number_input("Lowest Core Temperature :", value = 36.60, format="%.2f")
            inp['d1_mbp_min'] = st.number_input("Lowest Mean BP :", value = 68.00, format="%.2f")
            inp['d1_mbp_noninvasive_min'] = st.number_input("Lowest Mean BP(non-invasively measured) :", value = 68.00, format="%.2f")
            inp['d1_mbp_invasive_min'] = st.number_input("Lowest Mean BP(invasively measured) :", value = 73.00, format="%.2f")


        submit = st.form_submit_button("Predict Patient Survival")

    if submit:
        if inp['ventilated_apache']=='No':
            inp['ventilated_apache']=0
        else:
            inp['ventilated_apache']=1

        df = pd.DataFrame.from_dict([inp])
        X,pred = get_prediction(data=df, model=nn)

        st.markdown("""<style> .big-font { font-family:sans-serif; color: #1D7AA7 ; font-size: 30px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">{pred} chances of survival is predicted for the patient.</p>', unsafe_allow_html=True)
        #st.write(f" => {pred} is predicted. <=")

        p, shap_values = explain_model_prediction(X,nn,features)
        
        st.subheader('Extent of factors affecting Patient Survival')
        st_shap(p)
    

if __name__ == '__main__':
    main()