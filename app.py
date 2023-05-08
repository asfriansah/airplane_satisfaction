import streamlit as st
import pickle
import numpy as np

def main():
    background = """<div style = 'background-colour:black'; padding:13px>
                    <h2 style = 'text-align: center'>Prediction of Passenger Satisfaction</h2>
                    </div>"""
    st.markdown(background, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    cust = left.selectbox('Customer Type',('Loyal Customer','Disloyal Customer'))
    cls = left.selectbox('Class Type',('Economy','Economy Plus','Business'))
    gender = left.selectbox('Gender',('Male','Female'))
    travel = left.selectbox('Type Of Travel',('Personal Travel','Business Travel'))
    checkin = right.selectbox('Checkin Satisfaction',('0','1','2','3','4','5'))
    age = right.number_input('Age',0,100)
    flight = right.number_input('Flight Distance In KM',0,10000)
    arrival = right.number_input('Arrival Delay In Minutes',0,5000)
    button = st.button('Predict')

    if button:
        result = predict(cust,cls,gender,int(age),travel,int(flight),checkin,int(arrival))
        st.success(f'The passenger is {result}')

st.markdown("""<style>
                    [data-testid=stSidebar] {
                        background-color: #8b8bad;
                    }
                </style>
                """, unsafe_allow_html=True)
with st.sidebar:
    st.subheader('About')
    desc = """<div style="text-align: justify;">
                    This is an application for predicting whether airplane passengers are satisfied
                    or dissatisfied. Predictions using the XGBoost algorithm model with an accuracy of 81,6%
              </div>"""
    st.markdown(desc, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")

    with col2:
        st.image('model/plane.png',width=200)

    with col3:
        st.write("")
    
    title_alignment = """<h4 style = 'text-align: center'>created 2023 by andri asfriansah</h4>
                         <h4 style = 'text-align: center'>DASI (Drajatul Khair-Andri Asfriansah-
                         Safira Khairunnisa-Inggil Bagas)</h4>"""
    st.markdown(title_alignment, unsafe_allow_html=True)

    st.image('model/dslogo.png')

with open('model/scaler.pkl','rb') as file:
    scal = pickle.load(file)

with open('model/xgboost1.pkl','rb') as file:
    model = pickle.load(file)

def predict(customer,clas,gender,age,travel,flight,checkin,arrival):
    cust = lambda a : 0 if (a == 'Disloyal Customer') else 1
    cust = cust(customer)
    cust = 0 if customer == 'Disloyal Customer' else 1
    cls = 0 if clas == 'Economy' else 1 if clas == 'Economy Plus' else 2
    sex = 0 if gender == 'Male' else 1
    trav = 0 if travel == 'Personal Travel' else 1
        
    predictor = np.array([cust,cls,sex,age,trav,flight,checkin,arrival])
    predictor1 = predictor.reshape(1, -1) 
    predictor1 = predictor1.astype(int)
    predictor_scal = scal.transform(predictor1)
    prediction = model.predict(predictor_scal)
    verdict = 'Dissatisfied' if prediction == 0 else 'Satisfied'
    return verdict


if __name__ == "__main__":
    main()
