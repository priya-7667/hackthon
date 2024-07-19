import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from prophet import Prophet

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

def generate_sample_data():
    np.random.seed(0)
    date_rng = pd.date_range(start="2023-01-01", periods=180, freq='D')
    data = []
    
    for date in date_rng:
        for _ in range(24):  # 24 entries per day
            time = date + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            consumption = np.random.randint(200, 400)
            cost = np.random.uniform(50, 100)
            co2_emissions = np.random.uniform(100, 200)
            data.append([time, consumption, cost, co2_emissions])
    
    df = pd.DataFrame(data, columns=["DateTime", "Consumption (kWh)", "Cost ($)", "CO2 Emissions (kg)"])
    df.sort_values("DateTime", inplace=True)  # Ensure data is sorted by datetime
    return df

def customize_plot(fig):
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_font=dict(color='white'),
        legend_title=dict(font=dict(color='white')),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    return fig

def consumer_dashboard():
    st.title("Consumer Dashboard")
    st.subheader("Energy Consumption Data")

    df = generate_sample_data()

    # Date range filter
    st.subheader("Filter Data")
    start_date = st.date_input("Start date", df["DateTime"].min().date())
    end_date = st.date_input("End date", df["DateTime"].max().date())

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        mask = (df['DateTime'] >= pd.to_datetime(start_date)) & (df['DateTime'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]

        st.write(filtered_df)

        # Visualizations
        st.subheader("Visualizations")

        fig1 = px.line(filtered_df, x="DateTime", y="Consumption (kWh)", title="Hourly Energy Consumption")
        fig1 = customize_plot(fig1)
        st.plotly_chart(fig1)

        fig2 = px.bar(filtered_df, x="DateTime", y="Cost ($)", title="Hourly Energy Cost")
        fig2 = customize_plot(fig2)
        st.plotly_chart(fig2)

        fig3 = px.scatter(filtered_df, x="DateTime", y="CO2 Emissions (kg)", title="Hourly CO2 Emissions")
        fig3 = customize_plot(fig3)
        st.plotly_chart(fig3)

        # Future prediction using Prophet
        st.subheader("Future Prediction")

        # Prepare data for Prophet
        df_prophet = filtered_df.rename(columns={"DateTime": "ds", "Consumption (kWh)": "y"})
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=24*7, freq='H')  # Predict next 7 days
        forecast = model.predict(future)

        fig4 = px.line(forecast, x="ds", y="yhat", title="Future Energy Consumption Prediction")
        fig4 = customize_plot(fig4)
        st.plotly_chart(fig4)

        # Interactive heatmap
        st.subheader("Interactive Heatmap")
        heatmap_data = filtered_df.pivot(index='DateTime', columns='Cost ($)', values='Consumption (kWh)')
        fig5 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis'
        ))
        fig5 = customize_plot(fig5)
        st.plotly_chart(fig5)

def prosumer_dashboard():
    st.title("Prosumer Dashboard")
    st.subheader("Energy Production Data")

    df = generate_sample_data()

    # Date range filter
    st.subheader("Filter Data")
    start_date = st.date_input("Start date", df["DateTime"].min().date())
    end_date = st.date_input("End date", df["DateTime"].max().date())

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        mask = (df['DateTime'] >= pd.to_datetime(start_date)) & (df['DateTime'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]

        st.write(filtered_df)

        # Visualizations
        st.subheader("Visualizations")

        fig1 = px.line(filtered_df, x="DateTime", y="Consumption (kWh)", title="Hourly Energy Production")
        fig1 = customize_plot(fig1)
        st.plotly_chart(fig1)

        fig2 = px.bar(filtered_df, x="DateTime", y="Cost ($)", title="Hourly Revenue")
        fig2 = customize_plot(fig2)
        st.plotly_chart(fig2)

        fig3 = px.scatter(filtered_df, x="DateTime", y="CO2 Emissions (kg)", title="Hourly CO2 Savings")
        fig3 = customize_plot(fig3)
        st.plotly_chart(fig3)

        # Future prediction using Prophet
        st.subheader("Future Prediction")

        # Prepare data for Prophet
        df_prophet = filtered_df.rename(columns={"DateTime": "ds", "Consumption (kWh)": "y"})
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=24*7, freq='H')  # Predict next 7 days
        forecast = model.predict(future)

        fig4 = px.line(forecast, x="ds", y="yhat", title="Future Energy Production Prediction")
        fig4 = customize_plot(fig4)
        st.plotly_chart(fig4)

        # Interactive histogram
        st.subheader("Interactive Histogram")
        fig5 = px.histogram(filtered_df, x="Consumption (kWh)", nbins=30, title="Production Distribution")
        fig5 = customize_plot(fig5)
        st.plotly_chart(fig5)

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
