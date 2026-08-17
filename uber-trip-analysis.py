import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Uber Trip Data Analysis", layout="wide")

st.title("Uber Trip Data Analysis Dashboard")
st.write("Statistical Modelling and Machine Learning Project")
st.write("This dashboard analyzes daily trips, peak hours, average passengers, and ride demand.")


df = pd.read_csv(r"C:\Users\admin\Downloads\uber_trip_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df["DayOfWeek"] = df["Weekday"]

if st.checkbox("Show raw data"):
    st.dataframe(df.head(50))

st.write("Total trips in dataset:", df.shape[0])


st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trips", len(df))
col2.metric("Avg Passengers", round(df["Passengers"].mean(), 2))
col3.metric("Avg Distance (km)", round(df["Distance"].mean(), 2))
col4.metric("Avg Fare (Rs)", round(df["Fare"].mean(), 2))


col1, col2 = st.columns(2)
with col1:

    st.header("1. Daily Trips Trend")

    daily_trips = df.groupby(df["Date"].dt.date).size()
    st.line_chart(daily_trips)


with col2:

    st.header("2. Peak Hour Analysis")

    hourly_trips = df.groupby("Hour").size()
    st.bar_chart(hourly_trips)

    peak_hour = hourly_trips.idxmax()
    st.write("Peak hour is:", peak_hour, ":00 with", hourly_trips.max(), "trips")


with col1:    
    st.header("3. Ride Demand by Pickup Location")

    top_locations = df["Pickup Location"].value_counts().head(10)
    st.bar_chart(top_locations)

with col2:
        
    st.header("4. Ride Demand by Day of Week")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_trips = df["DayOfWeek"].value_counts().reindex(day_order)
    st.bar_chart(day_trips)

with col1:    
    st.header("5. Passenger Distribution")

    passenger_counts = df["Passengers"].value_counts()
    fig, ax = plt.subplots(figsize= (4,3))
    ax.pie(passenger_counts, labels=passenger_counts.index, autopct= "%1.1f%%")
    ax.set_title("Passenger Distribution")
    st.pyplot(fig, width="content")


with col2:    
    st.header("6. Fare vs Distance")

    fig, ax = plt.subplots(figsize= (4,3))
    ax.scatter(df["Distance"], df["Fare"], alpha=0.3)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Fare (Rs)")
    ax.set_title("Fare vs Distance")
    st.pyplot(fig, width="content")

correlation = df["Distance"].corr(df["Fare"])
st.write("Correlation between Distance and Fare:", round(correlation, 2))


st.header("Summary")

st.write("- Peak hour of the day is around", peak_hour, ":00")
st.write("- Busiest pickup location is", top_locations.idxmax())
st.write("- Busiest day of the week is", day_trips.idxmax())
st.write("- Average passengers per trip is", round(df["Passengers"].mean(), 2))