import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

def load_model(model_path):
    clf = pd.read_pickle(model_path)
    return clf

def base_data(data_path):
    data = pd.read_csv(data_path)
    return data

st.title("Hotel Booking Cancellation Predictor")
st.write("Enter the data about the booking to confirm its cancellation probability")

model_path = 'models/classifier.pkl'
data_path = 'data/processed/clean_data.csv'

clf = load_model(model_path)
data = base_data(data_path)

no_of_adults = st.number_input("Number of Adults", min_value = 1, max_value = 4, value = 1)
no_of_children = st.number_input("Number of Children", min_value = 0, max_value = 4, value = 0)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value = 0, max_value = 10, value = 0)
no_of_week_nights = st.number_input("Number of Week Nights", min_value = 0, max_value = 10, value = 0)
type_of_meal_plan = st.selectbox("Type of Meal Plan", list(data['type_of_meal_plan'].unique()))
required_car_parking_space = st.selectbox("Required Car Parking Space", ('Yes', 'No'))
room_type_reserved = st.selectbox("Room Type Reserved", list(data['room_type_reserved'].unique()))
lead_time = st.number_input("Lead Time", min_value=1, max_value=365, value = 1)
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.selectbox("Arrival Day", list(range(1, 32)))
market_segment_type = st.selectbox("Market Segment", list(data['market_segment_type'].unique()))
repeated_guest = st.selectbox("Repeated Guest", ('Yes', 'No'))
no_of_previous_cancellations = st.number_input("Previous Canceled Bookings", min_value = 0, max_value = 50, value = 0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Successful Bookings", min_value = 0, max_value = 50, value = 0)
avg_price_per_room = st.slider("Average Room Price", 0.00, 600.00)
no_of_special_requests = st.slider("Special Requests", 1, 5)

input_features = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'type_of_meal_plan': type_of_meal_plan,
    'required_car_parking_space': required_car_parking_space,
    'room_type_reserved': room_type_reserved,
    'lead_time': lead_time,
    'arrival_month': arrival_month,
    'arrival_date': arrival_date,
    'market_segment_type': market_segment_type,
    'repeated_guest': repeated_guest,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests
}

input_df = pd.DataFrame([input_features])
input_df['required_car_parking_space'] = input_df['required_car_parking_space'].map({'Yes': 1, 'No': 0})
input_df['repeated_guest'] = input_df['repeated_guest'].map({'Yes': 1, 'No': 0})

with st.container():
    st.write("")

    if st.button("Predict"):
        pred = clf['model'].predict_proba(input_df[clf['features']])[:,1]
        probability = float(pred) * 100

        if pred > 0.47:
            st.markdown("#### **Canceled**")
            st.write(f"This booking has a {probability:.2f}% probability of cancellation.")
            img_path = 'docs/xBGsTPh1nZRnJE4EIAij--3--ho4oj.jpg'
            st.image(img_path, use_column_width=True)
        else:
            st.markdown("#### **Not Canceled**")
            st.write(f"This booking has a {probability:.2f}% probability of cancellation.")
            img_path = 'docs/NjsCPadTLHyoNtq3hQmG--2--jeyr4.jpg'
            st.image(img_path, use_column_width=True)