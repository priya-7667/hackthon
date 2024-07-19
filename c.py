import streamlit as st
import pandas as pd
import numpy as np
import plotly as px
from sklearn.linear_model import LinearRegression
from datetime import datetime

# User credentials
USER = "manager"
PASSWORD = "password"

def login():
    st.title("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    user_type = st.selectbox("User Type", ["consumer", "prosumer"], key="login_user_type")

    if st.button("Login", key="login_button"):
        if username == USER and password == PASSWORD:
            st.session_state.logged_in = True
            st.session_state.user_type = user_type
        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_type" not in st.session_state:
    st.session_state.user_type = None

def consumer_dashboard():
    st.title("Consumer Dashboard")
    st.subheader("Energy Consumption Data")

    # Sample data for energy consumption
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=180, freq='D'),
        "Consumption (kWh)": np.random.randint(200, 400, size=180),
        "Cost ($)": np.random.uniform(50, 100, size=180),
        "CO2 Emissions (kg)": np.random.uniform(100, 200, size=180)
    }
    df = pd.DataFrame(data)

    # Date range filter
    st.subheader("Filter Data")
    start_date = st.date_input("Start date", df["Date"].min())
    end_date = st.date_input("End date", df["Date"].max())

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]

        st.write(filtered_df)

        # Visualizations
        st.subheader("Visualizations")
        
        fig1 = px.line(filtered_df, x="Date", y="Consumption (kWh)", title="Daily Energy Consumption")
        st.plotly_chart(fig1)

        fig2 = px.bar(filtered_df, x="Date", y="Cost ($)", title="Daily Energy Cost")
        st.plotly_chart(fig2)

        fig3 = px.scatter(filtered_df, x="Date", y="CO2 Emissions (kg)", title="Daily CO2 Emissions")
        st.plotly_chart(fig3)

        # Future prediction (simple linear regression for example)
        st.subheader("Future Prediction")

        X = np.arange(len(filtered_df)).reshape(-1, 1)  # Days as numbers
        y = filtered_df["Consumption (kWh)"].values

        model = LinearRegression()
        model.fit(X, y)
        future_days = np.arange(len(filtered_df) + 30).reshape(-1, 1)  # Predict next 30 days
        future_consumption = model.predict(future_days)

        future_df = pd.DataFrame({
            "Date": pd.date_range(start=filtered_df["Date"].max() + pd.Timedelta(days=1), periods=30, freq='D'),
            "Predicted Consumption (kWh)": future_consumption[-30:]
        })

        fig4 = px.line(future_df, x="Date", y="Predicted Consumption (kWh)", title="Future Energy Consumption Prediction")
        st.plotly_chart(fig4)

def prosumer_dashboard():
    st.title("Prosumer Dashboard")
    st.subheader("Energy Production Data")

    # Sample data for energy production
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=180, freq='D'),
        "Production (kWh)": np.random.randint(300, 500, size=180),
        "Revenue ($)": np.random.uniform(80, 150, size=180),
        "CO2 Savings (kg)": np.random.uniform(150, 250, size=180)
    }
    df = pd.DataFrame(data)

    # Date range filter
    st.subheader("Filter Data")
    start_date = st.date_input("Start date", df["Date"].min())
    end_date = st.date_input("End date", df["Date"].max())

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]

        st.write(filtered_df)

        # Visualizations
        st.subheader("Visualizations")

        fig1 = px.line(filtered_df, x="Date", y="Production (kWh)", title="Daily Energy Production")
        st.plotly_chart(fig1)

        fig2 = px.bar(filtered_df, x="Date", y="Revenue ($)", title="Daily Revenue")
        st.plotly_chart(fig2)

        fig3 = px.scatter(filtered_df, x="Date", y="CO2 Savings (kg)", title="Daily CO2 Savings")
        st.plotly_chart(fig3)

        # Future prediction (simple linear regression for example)
        st.subheader("Future Prediction")

        X = np.arange(len(filtered_df)).reshape(-1, 1)  # Days as numbers
        y = filtered_df["Production (kWh)"].values

        model = LinearRegression()
        model.fit(X, y)
        future_days = np.arange(len(filtered_df) + 30).reshape(-1, 1)  # Predict next 30 days
        future_production = model.predict(future_days)

        future_df = pd.DataFrame({
            "Date": pd.date_range(start=filtered_df["Date"].max() + pd.Timedelta(days=1), periods=30, freq='D'),
            "Predicted Production (kWh)": future_production[-30:]
        })

        fig4 = px.line(future_df, x="Date", y="Predicted Production (kWh)", title="Future Energy Production Prediction")
        st.plotly_chart(fig4)

def display_dashboard():
    if st.session_state.user_type == "consumer":
        consumer_dashboard()
    elif st.session_state.user_type == "prosumer":
        prosumer_dashboard()
    else:
        st.error("Invalid user type")

def display_erc_management():
    st.title("ERC Management")

    # Sample data for existing ERCs
    if 'ercs' not in st.session_state:
        st.session_state.ercs = [
            {"name": "Community 1", "users": ["User A", "User B"], "type": "consumer"},
            {"name": "Community 2", "users": ["User C", "User D"], "type": "prosumer"},
        ]
    ercs = st.session_state.ercs

    # Display existing ERCs
    st.subheader("Existing ERCs")
    for idx, erc in enumerate(ercs):
        st.write(f"Name: {erc['name']}")
        st.write(f"Users: {', '.join(erc['users'])}")
        st.write(f"Type: {erc['type']}")
        st.write("---")

    # Create a new ERC
    st.subheader("Create a New ERC")
    new_erc_name = st.text_input("ERC Name", key="new_erc_name")
    new_erc_type = st.selectbox("Type", ["consumer", "prosumer"], key="new_erc_type")

    if st.button("Create ERC", key="create_erc_button"):
        ercs.append({"name": new_erc_name, "users": [], "type": new_erc_type})
        st.session_state.ercs = ercs
        st.success(f"ERC '{new_erc_name}' created successfully")

    # Add a new user to an ERC
    st.subheader("Add a New User to an ERC")
    erc_to_add_user = st.selectbox("Select ERC", [erc["name"] for erc in ercs], key="erc_to_add_user")
    new_user_name = st.text_input("New User Name", key="new_user_name")

    if st.button("Add User", key="add_user_button"):
        for erc in ercs:
            if erc["name"] == erc_to_add_user:
                erc["users"].append(new_user_name)
                st.session_state.ercs = ercs
                st.success(f"User '{new_user_name}' added to ERC '{erc_to_add_user}'")

    # Change user type
    st.subheader("Change User Type")
    erc_to_change_user_type = st.selectbox("Select ERC for User Type Change", [erc["name"] for erc in ercs], key="erc_to_change_user_type")
    user_to_change_type = st.text_input("User to Change Type", key="user_to_change_type")
    new_user_type = st.selectbox("New Type", ["consumer", "prosumer"], key="change_user_new_type")

    if st.button("Change User Type", key="change_user_type_button"):
        for erc in ercs:
            if erc["name"] == erc_to_change_user_type:
                if user_to_change_type in erc["users"]:
                    erc["type"] = new_user_type
                    st.session_state.ercs = ercs
                    st.success(f"User '{user_to_change_type}' in ERC '{erc_to_change_user_type}' changed to '{new_user_type}'")

if not st.session_state.logged_in:
    login()
else:
    display_dashboard()
    display_erc_management()
