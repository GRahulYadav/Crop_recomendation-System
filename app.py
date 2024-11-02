# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model (assumes 'model.pkl' is the trained model file)
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Streamlit application layout
# st.title("Crop Recommendation System")

# # Input fields for user data
# nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0)
# phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0)
# potassium = st.number_input("Potassium (kg/ha)", min_value=0.0)
# temperature = st.number_input("Temperature (°C)", min_value=0.0)
# humidity = st.number_input("Humidity (%)", min_value=0.0)
# ph = st.number_input("pH", min_value=0.0)
# rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# # Button to make prediction
# if st.button("Predict Crop"):
#     # Prepare data for prediction
#     input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
#     # Make prediction
#     prediction = model.predict(input_data)
    
#     # Display the prediction
#     st.success(f"Recommended Crop: {prediction[0]}")
import streamlit as st
import pickle
import numpy as np

# Load the trained model (assumes 'model.pkl' is the trained model file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Streamlit application layout
st.title("Crop Recommendation System")

# Input fields for user data
nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0)
phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0)
potassium = st.number_input("Potassium (kg/ha)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Button to make prediction
if st.button("Predict Crop"):
    # Prepare data for prediction
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Make prediction
    predicted_class = model.predict(input_data)

    # Decode the predicted class back to the original label
    predicted_crop = le.inverse_transform(predicted_class)

    # Display the prediction
    st.success(f"Recommended Crop: {predicted_crop[0]}")
