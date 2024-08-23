import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

def main():
    page_icon_url = "https://github.com/user-attachments/assets/69df246e-ddca-4992-826a-e44889ddd698"
    st.set_page_config(
        page_title="Customer Loan Predictor",
        page_icon=page_icon_url,
        layout="wide",
    )
    st.subheader("⏩ Customer **Loan Predictor** | _By Gopi_ ")

    age = list(range(18, 96))
    day = list(range(1,32))
    job = [
        "MANAGEMENT",
        "TECHNICIAN",
        "ENTREPRENEUR",
        "BLUECOLLAR",
        "OTHER",
        "RETIRED",
        "ADMIN",
        "SERVICES",
        "SELFEMPLOYED",
        "UNEMPLOYED",
        "HOUSEMAID",
        "STUDENT",
    ]
    marital = ["MARRIED", "SINGLE", "DIVORCED"]

    education_qual = ["TERTIARY", "SECONDARY", "OTHER", "PRIMARY"]

    call_type = ["OTHER", "CELLULAR", "TELEPHONE"]
    
    columns_to_scale = ["age", "day", "mon", "dur", "job"]

    mon = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]

    month_dict = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    prev_outcome = ["OTHER", "FAILURE", "OTHER", "SUCCESS"]

    job_dict = {
        "MANAGEMENT": 0,
        "TECHNICIAN": 1,
        "ENTREPRENEUR": 2,
        "BLUECOLLAR": 3,
        "OTHER": 4,
        "RETIRED": 5,
        "ADMIN": 6,
        "SERVICES": 7,
        "SELFEMPLOYED": 8,
        "UNEMPLOYED": 9,
        "HOUSEMAID": 10,
        "STUDENT": 11,
    }

    marital_dict = {"MARRIED": 0, "SINGLE": 1, "DIVORCED": 2}

    education_qual_dict = {"TERTIARY": 0, "SECONDARY": 1, "OTHER": 2, "PRIMARY": 3}

    call_type_dict = {"OTHER": 0, "CELLULAR": 1, "TELEPHONE": 2}

    prev_outcome_dict = {"OTHER": 0, "FAILURE": 1, "OTHER": 2, "SUCCESS": 3}

    outcome_dict = {"NO": 0, "YES": 1}
    
        # Custom CSS for the submit button
    st.markdown(
        """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 80px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: 2px solid #4CAF50;
    }
    .stButton button:hover {
        background-color: white;
        color: #4CAF50;
    }
    .prediction_win{
        font-size: 50px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-top: 20px;
    }
    .prediction_lost{
        font-size: 50px;
        font-weight: bold;
        color: #eb0707;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    st.info("*Values Refered are based on given Data")
    with st.form("flat_price_predictor_form"):
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        
        with col1:
            age = st.selectbox("Select Age",options=age)
            
            job = st.selectbox("Select Occupation",options=job)
            
            marital = st.selectbox("Select Marital Status",options=marital)
            
            education_qual = st.selectbox("Select Educational Qualification",education_qual)
            
            call_type = st.selectbox("Select Call Type",options=call_type)
        
        with col3:
            
            day = st.selectbox("Select Day",options=day) 
            
            mon = st.selectbox("Select Month",options=mon)
            
            dur = st.number_input("Enter Duration",min_value=0,value=0)
            
            num_calls = st.number_input("Enter Number of Calls",min_value=0,value=0)
            
            prev_outcome =st.selectbox("Select Previous Outcome",options=prev_outcome)
            
            st.write(" ")

            submitted = st.form_submit_button("Submit")
            
        if submitted:
            
            try:
                model_path = os.path.join(os.path.dirname(__file__), "Models", "insurance_customer_predictor.pkl")
                scaler_path =os.path.join(os.path.dirname(__file__), "Models", "scaler.pkl")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                    age = age 
                    
                    job = job_dict[job]
                    
                    marital = marital_dict[marital]
                    
                    education_qual = education_qual_dict[education_qual]
                    
                    call_type =call_type_dict[call_type]
                    
                    day=day
                    
                    mon = month_dict[mon]
                    
                    dur = dur
                    
                    num_calls = num_calls
                    
                    prev_outcome = prev_outcome_dict[prev_outcome]
                    
                    user_data = np.array(
                        [
                            [
                                age,
                                job,
                                marital,
                                education_qual,
                                call_type,
                                day,
                                mon,
                                dur,
                                num_calls,
                                prev_outcome,
                            ]
                        ]
                    )
                    
                    
                    user_data_df = pd.DataFrame(user_data, columns=["age", "job", "marital", "education_qual", "call_type", "day", "mon", "dur", "num_calls", "prev_outcome"])
                    
                    # Scale the user data
                    user_data_scaled = user_data_df.copy()
                    
                    with open(scaler_path,"rb") as sc:
                        scaler = pickle.load(sc)
                        
                    user_data_scaled[columns_to_scale] = scaler.transform(user_data_df[columns_to_scale])
                    
                    y_prediction = model.predict(user_data_scaled)
                    outcome = y_prediction[0]
                    Outcome = "YES" if outcome == 1 else "NO"
                    if outcome == 1:
                        st.markdown(
                            f"<div class='prediction_win'>{Outcome}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='prediction_lost'>{Outcome}</div>",
                            unsafe_allow_html=True,
                        )
                    
                    
            except ValueError as e:
                st.warning(e)
                st.warning("Please Provide Valid Details")
            except Exception as e:
                st.warning(e)
         
            


if __name__ == "__main__":
    main()
