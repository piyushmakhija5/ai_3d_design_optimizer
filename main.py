import os
from dotenv import load_dotenv
import openai
import streamlit as st
from gan_model import GAN3DModelGenerator  # Import the modularized GAN class
from dqn_optimizer import run_dqn_optimization  # Import DQN-related function

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Correctly set API key

# Function to call GPT-4o-mini or any other OPENAI model and extract 3D design parameters
def extract_design_params(text):
    messages = [
        {"role": "system", "content": "You are an assistant that extracts 3D design parameters from user input. Your response should be in JSON format with the following structure: {'Type': '', 'Load Capacity': '', 'Features': ''}. Populate the fields with the appropriate values from the input."},
        {"role": "user", "content": f"Extract the design parameters from this input: '{text}'."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=messages,
        temperature=0.7
    )
    output = response['choices'][0]['message']['content'].strip()
    return output

# Streamlit Interface
st.title('AI-Assisted 3D Design Optimization')
st.subheader('Enter your design requirements below:')

# Initialize session state variables
if "gan_model" not in st.session_state:
    st.session_state.gan_model = None

if "design_params" not in st.session_state:
    st.session_state.design_params = None

if "feedback" not in st.session_state:
    st.session_state.feedback = None

if "optimized_design" not in st.session_state:
    st.session_state.optimized_design = None

# User input
user_input = st.text_area('Design Requirements', placeholder='E.g. I need a bracket that can support 20kg with two mounting holes')

if st.button('Submit'):
    if user_input:
        st.write(f'Processing your input: {user_input}')
        
        # Extract design parameters
        st.session_state.design_params = extract_design_params(user_input)
        st.subheader('Extracted Design Parameters:')
        st.json(st.session_state.design_params)
        
        # Generate 3D model using GAN
        st.write("Generating initial 3D model based on the extracted parameters...")
        st.session_state.gan_model = GAN3DModelGenerator()
        st.session_state.gan_model.train()  # Train the model (This could take time depending on your system)
        st.session_state.gan_model.generate_and_visualize()  # Generate and visualize the model

# Collect user feedback for optimization
if st.session_state.design_params:
    feedback = st.text_input("Initial Design generated. How would you like to optimize it?", value=st.session_state.feedback or "")
    if feedback:
        st.session_state.feedback = feedback

        if st.button('Run Optimization'):
            st.write(f"User Feedback: {feedback}")
            
            # Run DQN Optimization based on user feedback
            st.write("Running DQN optimization based on user feedback...")
            st.session_state.optimized_design = run_dqn_optimization(feedback, st.session_state.design_params)

            st.write("DQN Optimization complete. Design parameters have been adjusted based on user feedback.")
            st.json(st.session_state.optimized_design)

# Option to download the final 3D output design file in STL format
if st.session_state.optimized_design:
    if st.button('Download Optimized Design'):
        st.write("Downloading optimized 3D design as STL... (placeholder functionality)")
