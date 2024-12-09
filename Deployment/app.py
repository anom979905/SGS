import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import gen_labels, preprocess, model_arc  # Assuming these are valid functions
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["OPEN_AI_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "hi\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Hi there! What can I do for you today? \n",
      ],
    },
  ]
)
response = chat_session.send_message("waht is the carbon emission percentage of the metal waste")
print(response.text)

###########################################################################################################################################


# Hide Streamlit's default menu and footer, including the "Deploy" button
st.set_page_config(page_title="Smart Garbage Classification", page_icon=":wastebasket:")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the labels
labels = gen_labels()

# HTML section for header and images    
html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
      <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d7">Smart </span>Garbage</h1></center>
      <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
      <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
  </div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

# User input for image upload method
opt = st.selectbox("How do you want to upload the image for classification?", ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None  # Initialize image variable

# Image uploading
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
    img_url = st.text_input('Enter the Image Address')
    if img_url:
        try:
            image = Image.open(urllib.request.urlopen(img_url))
        except Exception as e:
            st.error(f"Error fetching image: {e}")

if image:
    st.image(image, width=300, caption='Uploaded Image')

    # Predict button
    if st.button('Classify'):
        try:
            img = preprocess(image)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # **Keras model** for image classification
            keras_model = model_arc()  # This is for image classification
            if keras_model is None:
                st.error("Model initialization failed.")
            else:
                keras_model.load_weights("../weights/modelnew.h5")  # Update path as needed

                # Perform the image prediction using Keras model
                prediction = keras_model.predict(img)
                predicted_label = labels[np.argmax(prediction[0])]
                response1 = chat_session.send_message(f'Give the information of CO2 emmission ,recycaling type in what recycle prodcut,how much harmfull this to our environemtn  of  "{predicted_label}", only give the information dont reply this way "tricky to give a precise percentage for glass waste carbon emissions because it varies depending on factors like:"')
                st.info(f'{response1.text}')

               

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.info("Please upload an image to classify.")


