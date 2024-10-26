# AI-Assisted 3D Design Optimization

## Overview

This project is an AI-assisted platform for 3D design optimization. The system uses NLP to extract design requirements, GAN to generate initial 3D models, and DQN to optimize these models based on user feedback. The goal is to create an intuitive and efficient workflow for generating and refining 3D designs.

## Features

- **NLP Interface**: Uses OpenAI's GPT-4o-mini model to extract key design parameters from natural language input.
- **3D Model Generation**: Utilizes a Generative Adversarial Network (GAN) to create a basic 3D model based on the extracted design parameters.
- **Design Optimization**: Implements a Deep Q-Network (DQN) to adjust and optimize the design based on user feedback.

## Requirements

- Python 3.8 or higher
- Streamlit
- PyTorch
- OpenAI API

## Setup Instructions

Follow these steps to set up and run the project:

1. **Add .env File**
   - Create a `.env` file in the base folder and add your OpenAI API key:
     ```
     OPENAI_API_KEY='your_openai_api_key_here'
     ```

2. **Create Virtual Environment**
   - Use Conda or Python to create a virtual environment:
     ```
     conda create -n ai_3d_design_optimizer python=3.8
     conda activate ai_3d_design_optimizer
     ```
     or
     ```
     python -m venv <env-name>
     source <env-name>/bin/activate  # Linux/Mac
     <env-name>\Scripts\activate  # Windows
     ```

3. **Install Requirements**
   - Install all the required packages using `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

4. **Run Streamlit App**
   - Launch the Streamlit app:
     ```
     streamlit run main.py
     ```

## Workflow

1. **User Input**: Enter the design requirements in natural language.
2. **NLP Extraction**: The system extracts key parameters like type, load capacity, and features.
3. **3D Model Generation**: A GAN generates an initial 3D model based on the parameters.
4. **User Feedback and Optimization**: Provide feedback to optimize the design. The DQN optimizes the model accordingly.

## Potential Improvements

- **Better Dataset**: Use a dataset like ShapeNet for improved model quality.
- **Advanced Reinforcement Learning**: Implement PPO for more efficient optimization.
- **Enhanced Feedback**: Integrate computer vision to evaluate structural integrity.

