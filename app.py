import streamlit as st
import pandas as pd
import joblib
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def load_data(path):
    return pd.read_csv(path)


st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.subheader("Insert the booking info and predict the cancellation probability.")


model = joblib.load('models/classifier.pkl')
data = load_data("data/raw/Hotel Reservations.csv")

st.sidebar.header("Booking Info")
icon_path = "docs/img/booking.png"
st.logo(icon_path)
no_of_adults = st.sidebar.number_input("Number of Adults", min_value = 1, max_value = 4, value = 1)
no_of_children = st.sidebar.number_input("Number of Children", min_value = 0, max_value = 4, value = 0)
no_of_weekend_nights = st.sidebar.number_input("Number of Weekend Nights", min_value = 0, max_value = 10, value = 0)
no_of_week_nights = st.sidebar.number_input("Number of Week Nights", min_value = 0, max_value = 10, value = 0)
type_of_meal_plan = st.sidebar.selectbox("Type of Meal Plan", list(data['type_of_meal_plan'].sort_values().unique()))
required_car_parking_space = st.sidebar.selectbox("Required Car Parking Space", ('Yes', 'No'))
room_type_reserved = st.sidebar.selectbox("Room Type Reserved", list(data['room_type_reserved'].sort_values().unique()))
lead_time = st.sidebar.number_input("Lead Time", min_value=1, max_value=365, value = 1)
arrival_year = st.sidebar.selectbox("Arrival Year", list(range(2024, 2031)))
arrival_month = st.sidebar.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.sidebar.selectbox("Arrival Day", list(range(1, 32)))
market_segment_type = st.sidebar.selectbox("Market Segment", list(data['market_segment_type'].unique()))
repeated_guest = st.sidebar.selectbox("Repeated Guest", ('Yes', 'No'))
no_of_previous_cancellations = st.sidebar.number_input("Previous Canceled Bookings", min_value = 0, max_value = 50, value = 0)
no_of_previous_bookings_not_canceled = st.sidebar.number_input("Previous Successful Bookings", min_value = 0, max_value = 50, value = 0)
avg_price_per_room = st.sidebar.slider("Average Room Price", 80.00, 600.00)
no_of_special_requests = st.sidebar.slider("Special Requests", 1, 5)

input_features = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'type_of_meal_plan': type_of_meal_plan,
    'required_car_parking_space': required_car_parking_space,
    'room_type_reserved': room_type_reserved,
    'lead_time': lead_time,
    'arrival_year': arrival_year,
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
        prob = model.predict_proba(input_df)[:,1][0]
        probability = float(prob) * 100

        if prob > 0.47:
            st.markdown('#### **Canceled**')
            st.write(f"This booking has a {probability:.2f}% probability of cancellation")
            img_path = 'docs/img/xBGsTPh1nZRnJE4EIAij--3--ho4oj.jpg'
            st.image(img_path, use_container_width = True)
        else:
            st.markdown("#### **Not Canceled**")
            st.write(f"This booking has a {probability:.2f}% probability of cancellation.")
            img_path = 'docs/img/NjsCPadTLHyoNtq3hQmG--2--jeyr4.jpg'
            st.image(img_path, use_container_width = True)