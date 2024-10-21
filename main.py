import streamlit as st
import os
from dotenv import load_dotenv
import openai  # Import openai directly

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")  # Correctly set API key

# Function to call GPT-4o-mini (as specified) and extract 3D design parameters
def extract_design_params(text):
    # Chat completion model setup using gpt-4o-mini as per the curl request
    messages = [
        {"role": "system", "content": "You are an assistant that extracts 3D design parameters from user input. Your response should be in JSON format with the following structure: {'Type': '', 'Load Capacity': '', 'Features': ''}. Populate the fields with the appropriate values from the input."},
        {"role": "user", "content": f"Extract the design parameters from this input: '{text}'."}
    ]
    
    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=messages,
        temperature=0.7
    )

    # Extract the response content
    output = response['choices'][0]['message']['content'].strip()

    return output

# Streamlit Interface
st.title('AI-Assisted 3D Design Optimization')
st.subheader('Enter your design requirements below:')

# User input
user_input = st.text_area('Design Requirements', placeholder='E.g. I need a bracket that can support 20kg with two mounting holes')

if st.button('Submit'):
    if user_input:
        st.write(f'Processing your input: {user_input}')

        # Extract design parameters using GPT-4o-mini
        design_params = extract_design_params(user_input)

        st.subheader('Extracted Design Parameters:')
        st.json(design_params)
