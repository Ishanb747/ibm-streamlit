import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

class LinearRegression:
    def __init__(self, rate, iterations):
        self.rate = rate
        self.iterations = iterations
    
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.update_weights()
    
    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = -(2*(self.X.T).dot(self.Y - Y_prediction))/self.m
        db = -2*np.sum(self.Y - Y_prediction)/self.m
        self.w = self.w - self.rate*dw
        self.b = self.b - self.rate*db
    
    def predict(self, X):
        return X.dot(self.w) + self.b

# Change the working directory to the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

model_path = os.path.join(current_dir, 'car.sav')
try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {model_path}")
    loaded_model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    loaded_model = None

# Define the prediction function
def emission_prediction(input_data):
    test_input = np.array(input_data)
    test_input_reshaped = test_input.reshape(1, -1)

    # Make a prediction using the loaded model
    try:
        if loaded_model is not None:
            prediction = loaded_model.predict(test_input_reshaped)
            return prediction
        else:
            print("Model is not loaded properly.")
            return None
    except Exception as e:
        print(f"An error occurred while making the prediction: {e}")
        return None

# Streamlit app
img_path = os.path.join(current_dir, 'true.jpg')
add_bg_from_local(img_path)  # Replace with your image file path
st.title('Car Emission Prediction')

engine_size = st.number_input('Engine Size (L)')
num_cylinders = st.number_input('Number of Cylinders')
fuel_type = st.selectbox('Fuel Type', options=['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
fuel_consumption = st.number_input('Fuel Consumption(L/100 km)')
co2_rating = st.number_input('CO2 Rating')
smog_rating = st.number_input('Smog Rating')

fuel_type_dict = {'Gasoline': 0, 'Diesel': 1, 'Electric': 2, 'Hybrid': 3}
fuel_type_encoded = fuel_type_dict[fuel_type]

# Adjust input_data based on the model's expected number of features
input_data = [engine_size, num_cylinders, fuel_type_encoded, fuel_consumption, co2_rating, smog_rating]

if st.button('Predict'):
    prediction = emission_prediction(input_data)
    if prediction is not None:
        prediction_value = prediction[0]
        st.write(f'Predicted Emission: {prediction_value} g/km')
        
        if prediction_value > 180:
            st.write('High range')
            st.write('Vehicles producing over 180 grams of CO2 per kilometer fall into the high range. Typically, these include large SUVs, trucks, and performance cars. They have higher fuel consumption and significant environmental impacts, necessitating investment in cleaner technologies and stricter emission policies to reduce their carbon footprint.')
        elif 110 < prediction_value < 180:
            st.write('Medium range')
            st.write('Vehicles emitting between 110 and 180 grams of CO2 per kilometer are in the medium range. This category includes many conventional gasoline and diesel vehicles with moderate fuel efficiency. Despite efforts to reduce emissions, these vehicles contribute to urban pollution and greenhouse gas emissions. Eco-friendly driving practices can help mitigate their impact.')
        elif prediction_value < 110:
            st.write('Low range')
            st.write('Vehicles emitting less than 110 grams of CO2 per kilometer are considered safe. These include hybrids and electric vehicles, which significantly reduce environmental impact by utilizing advanced technologies and alternative energy sources. They are often supported by government incentives aimed at promoting cleaner transportation options and improving urban air quality.')
    else:
        st.write('Prediction could not be made due to an error with the model or input data.')