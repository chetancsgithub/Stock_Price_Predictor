import hashlib
import sqlite3
from datetime import date

import numpy as np
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function to check hashed passwords
def verify_password(password, hashed_password):
    return hash_password(password) == hashed_password

# Initialize SQLite database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        is_admin INTEGER DEFAULT 0
    )
''')
c.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        stock_symbol TEXT PRIMARY KEY
    )
''')
conn.commit()

# Registration function
def register_user(username, password, is_admin=False):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False  # User already exists
    hashed_password = hash_password(password)
    c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, hashed_password, int(is_admin)))
    conn.commit()
    return True  # User registered successfully

# Login function
def login_user(username, password):
    c.execute('SELECT password, is_admin FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    if data:
        return verify_password(password, data[0]), data[1] == 1
    return False, False

# Function to add a new stock
def add_stock(stock_symbol):
    c.execute('INSERT OR IGNORE INTO stocks (stock_symbol) VALUES (?)', (stock_symbol,))
    conn.commit()

# Function to get the list of stocks
def get_stocks():
    c.execute('SELECT stock_symbol FROM stocks')
    return [row[0] for row in c.fetchall()]

# Initialize default stocks
default_stocks = ["AAPL", "GOOG", "MSFT", "GME", "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATAPOWER.NS", "TSLA", "TSLA.NE",
                  "PNB.NS", "ADANIENT.NS", "ADANIPOWER.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "AMZN", "SBI", "YESBANK.NS",
                  "UNIONBANK.NS"]
for stock in default_stocks:
    add_stock(stock)

# Streamlit UI
st.markdown("""
<style>
    [data-testid="stAppViewContainer"]{
        background-image: url("https://images.pexels.com/photos/187041/pexels-photo-187041.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
        background-size: cover;
    }
</style>
""", unsafe_allow_html=True)

custom_color = "#FFFFFF"
st.markdown(f"<h1 style='color:{custom_color}'>Stock Prediction App</h1>", unsafe_allow_html=True)

# Initialize session state for username and password fields
if 'username_input' not in st.session_state:
    st.session_state['username_input'] = ''
if 'password_input' not in st.session_state:
    st.session_state['password_input'] = ''
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

# Function to handle login and registration
def auth():
    auth_mode = st.sidebar.selectbox("Choose Auth Mode", ["Login", "Register"])

    if auth_mode == "Register":
        st.sidebar.subheader("Register")
        new_user = st.sidebar.text_input("Username", key="register_username")
        new_password = st.sidebar.text_input("Password", type="password", key="register_password")
        is_admin = st.sidebar.checkbox("Register as admin", key="register_is_admin")
        if st.sidebar.button("Register"):
            if new_user and new_password:
                if register_user(new_user, new_password, is_admin):
                    st.sidebar.success("User registered successfully!")
                else:
                    st.sidebar.error("Username already taken. Please choose another.")
            else:
                st.sidebar.error("Please enter a valid username and password")

    if auth_mode == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            logged_in, is_admin = login_user(username, password)
            if logged_in:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = is_admin
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid username or password")

# Function to handle logout
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['is_admin'] = False
    st.session_state['username_input'] = ''
    st.session_state['password_input'] = ''
    st.sidebar.success("Logged out successfully!")

# Main app content
if st.session_state['logged_in']:
    st.sidebar.button("Logout", on_click=logout)
    st.markdown(f"<h1 style='color:{custom_color}'>Welcome, {st.session_state['username']}</h1>", unsafe_allow_html=True)
    
    if st.session_state['is_admin']:
        st.sidebar.subheader("Admin Panel")
        new_stock = st.sidebar.text_input("Add Stock Symbol", key="new_stock")
        if st.sidebar.button("Add Stock"):
            if new_stock:
                add_stock(new_stock)
                st.sidebar.success(f"Stock {new_stock} added successfully!")
            else:
                st.sidebar.error("Please enter a valid stock symbol")

    start = '2015-01-01'
    today = date.today().strftime("%Y-%m-%d")

    stocks = get_stocks()
    st.markdown(f"<style>.stSelectbox label {{color:{custom_color};}}</style>", unsafe_allow_html=True)
    selected_stocks = st.selectbox("Select dataset for prediction", stocks)

    st.markdown(
        f"""
        <style>
            .stSlider > div > div > div > div {{
                background-color: {custom_color} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<style>.stSlider label {{color:{custom_color};}}</style>", unsafe_allow_html=True)
    n_years = st.slider("Years of Prediction", 1, 7)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, start, today)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.markdown(f"<p style='color: white;'>Loading data...</p>", unsafe_allow_html=True)
    data = load_data(selected_stocks)
    data_load_state.text(st.markdown(f"<p style='color: white;'>Loading data...Done!</p>", unsafe_allow_html=True))

    st.markdown(f"<h1 style='color:{custom_color}'>Raw Data</h1>", unsafe_allow_html=True)
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_Close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.markdown(f"<h1 style='color:{custom_color}'>Forecast data</h1>", unsafe_allow_html=True)
    st.write(forecast.tail())

    st.markdown(f"<p style='color:{custom_color}'>forecast data</p>", unsafe_allow_html=True)
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.markdown(f"<p style='color:{custom_color}'>forecast components</p>", unsafe_allow_html=True)
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Calculate accuracy
    actual = data['Close']
    predicted = forecast['yhat'][:len(actual)]  # Align lengths for comparison
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy = 100 - mape

    st.markdown(f"<h1 style='color:{custom_color}'>Model Accuracy</h1>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color:{custom_color};'>Accuracy: <span style='color:{custom_color}'>{accuracy:.2f}%</span></p>", unsafe_allow_html=True)

else:
    auth()
    st.markdown("<h2 style='color:white;'>Please log in to use the app.</h2>", unsafe_allow_html=True)
