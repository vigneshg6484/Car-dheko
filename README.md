import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pickle
import os

def load_data():
    city_files = [f for f in os.listdir() if f.endswith('.xlsx')]
    city_dataframes = []
    
    for city_file in city_files:
        df = pd.read_excel(city_file)
        df['City'] = city_file.replace('.xlsx', '')
        city_dataframes.append(df)
    
    combined_data = pd.concat(city_dataframes, ignore_index=True)
    return combined_data

def preprocess_data(df):
    df = df.dropna()
    df['Mileage'] = df['Mileage'].str.replace(' kms', '').astype(int)
    df['Engine'] = df['Engine'].str.replace(' CC', '').astype(int)
    encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_columns = ['Make', 'Model', 'Fuel_Type', 'Transmission', 'City']
    encoded = encoder.fit_transform(df[categorical_columns])
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    df = pd.concat([df.drop(columns=categorical_columns), df_encoded], axis=1)
    scaler = StandardScaler()
    numerical_columns = ['Mileage', 'Engine', 'Year']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler, encoder

def train_model(df):
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, mse, r2

def save_model(model, scaler, encoder):
    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

st.title('Used Car Price Prediction')

if os.path.exists('car_price_model.pkl'):
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    make = st.selectbox('Make', ['Toyota', 'Honda', 'BMW'])
    model_name = st.text_input('Model Name')
    year = st.slider('Year', 2000, 2023, 2015)
    mileage = st.number_input('Mileage (in kms)', min_value=0)
    engine = st.number_input('Engine Capacity (in CC)', min_value=500)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    city = st.selectbox('City', ['Mumbai', 'Delhi', 'Bangalore'])
    
    input_data = pd.DataFrame([[make, model_name, year, mileage, engine, fuel_type, transmission, city]], 
                              columns=['Make', 'Model', 'Year', 'Mileage', 'Engine', 'Fuel_Type', 'Transmission', 'City'])
    input_encoded = encoder.transform(input_data[['Make', 'Model', 'Fuel_Type', 'Transmission', 'City']])
    input_scaled = scaler.transform(input_data[['Mileage', 'Engine', 'Year']])
    
    input_final = pd.concat([pd.DataFrame(input_scaled), pd.DataFrame(input_encoded)], axis=1)
    
    if st.button('Predict Price'):
        predicted_price = model.predict(input_final)
        st.write(f'Estimated Price: ₹{predicted_price[0]:,.2f}')
else:
    st.write('Training the model...')
    df = load_data()
    df, scaler, encoder = preprocess_data(df)
    model, mae, mse, r2 = train_model(df)
    st.write(f'Model Performance - MAE: {mae}, MSE: {mse}, R2: {r2}')
    save_model(model, scaler, encoder)
    st.write('Model trained and saved successfully!')
