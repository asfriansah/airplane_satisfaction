import streamlit as st
import pickle
import numpy as np

def main():
    background = """<div style = 'background-colour:black'; padding:13px>
                    <h2 style = 'colour:white'>Prediction of Passenger Satisfaction</h2>
                    </div>"""
    st.markdown(background, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    cust = left.selectbox('Customer Type',('Loyal Customer','Disloyal Customer'))
    cls = left.selectbox('Class Type',('Economy','Economy Plus','Business'))
    gender = left.selectbox('Gender',('Male','Female'))
    travel = left.selectbox('Type Of Travel',('Personal Travel','Business Travel'))
    checkin = right.selectbox('Checkin Satisfaction',('0','1','2','3','4','5'))
    age = right.number_input('Age')
    flight = right.number_input('Flight Distance In KM')
    arrival = right.number_input('Arrival Delay In Minutes')
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
                    or dissatisfied. Predictions using the XGBoost algorithm model with an accuracy of 82%
              </div>"""
    st.markdown(desc, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")

    with col2:
        st.image('model/plane.png',width=200)

    with col3:
        st.write("")
    
    title_alignment = """<h4 style = 'text-align: center'>created 2023 by andri asfriansah</h4>"""
    st.markdown(title_alignment, unsafe_allow_html=True)

with open('model/model2804.pkl','rb') as file:
    model = pickle.load(file)

def predict(customer,clas,gender,age,travel,flight,checkin,arrival):
    cust = lambda a : 0 if (a == 'Disloyal Customer') else 1
    cust = cust(customer)
    cust = 0 if customer == 'Disloyal Customer' else 1
    cls = 0 if clas == 'Economy' else 1 if clas == 'Economy Plus' else 2
    sex = 0 if gender == 'Male' else 1
    trav = 0 if travel == 'Personal Travel' else 1
    
    if age <= 10 :
        newage = 0
    elif age > 11 and age <= 20 :
        newage = 1
    elif age > 21 and age <= 30 :
        newage = 2
    elif age > 31 and age <= 40 :
        newage = 3
    elif age > 41 and age <= 50 :
        newage = 4
    elif age > 51 and age <= 60 :
        newage = 5
    elif age > 61 and age <= 70 :
        newage = 6
    else:
        newage = 7
   
    if arrival <= 180 :
        newarrival = 0
    elif arrival > 180 and arrival <= 360 :
        newarrival = 1
    elif arrival > 360 and arrival <= 540 :
        newarrival = 2
    elif arrival > 540 and arrival <= 720 :
        newarrival = 3
    elif arrival > 720 and arrival <= 900 :
        newarrival = 4
    elif arrival> 900 and arrival <= 1080 :
        newarrival = 5
    elif arrival > 1080 and arrival <= 1260 :
        newarrival = 6
    elif arrival > 1260 and arrival <= 1440 :
        newarrival= 7
    else:
        newarrival = 8
    
    predictor = np.array([cust,cls,sex,newage,trav,flight,checkin,newarrival])
    predictor1 = predictor.reshape(1, -1) 
    predictor1 = predictor1.astype(int)
    prediction = model.predict(predictor1)
    verdict = 'Dissatisfied' if prediction == 0 else 'Satisfied'
    return verdict


if __name__ == "__main__":
    main()
