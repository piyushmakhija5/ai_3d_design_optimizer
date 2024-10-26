# AI-Assisted 3D Design Optimization

## 1. Extracting Design Requirements with NLP

First, I used OpenAI's GPT-4o-mini model to extract design requirements from user input. Users describe their design needs in natural language, and the model extracts key attributes like type, load capacity, and features. This is done through prompt engineering, which ensures the output is in a structured format (JSON). I chose a pretained LLM  because of its ability to understand complex descriptions, making it easy for users to specify their design needs without needing technical knowledge. Choosing OpenAI's this specific model was a choice based on speed not the best fit. Other models from OpenAI, claude models, llama models or any other self-hosted LLM can also acheive similar results.

## 2. Generating 3D Models with GAN

Once the design parameters are extracted, I use a Generative Adversarial Network (GAN) to create a basic 3D model. The GAN has two main parts:

- **Generator**: Creates 3D models based on the input parameters.
- **Discriminator**: Evaluates how realistic the generated model is compared to known 3D models.

The generator and discriminator work together to improve the quality of the models over time. I used PyTorch to build the GAN because it provides flexibility and support for dynamic computations, making it suitable for training complex models like GANs.<br>
NOTE: Using a ShapeNet kind of 3D model dataset to train the GAN model would be really helpful. Current model output is Gibberish at best.

## 3. Optimizing Designs with DQN

After generating the initial design, I optimize it based on user feedback using a Deep Q-Network (DQN). The DQN helps adjust the design parameters to meet specific goals, such as reducing weight while keeping the strength. Here's how it works:

- The DQN learns a policy that maps the current state (design parameters) to actions (e.g., reduce weight).
- A **reward function** guides the DQN to make the right changes, rewarding actions that bring the design closer to user objectives.
- I use user feedback to adjust the rewards, making sure the optimization process aligns with user needs.
- A replay memory is used to make the learning process more stable and efficient.

## 4. Challenges and Improvements

Integrating the different components—NLP, GAN, and DQN—into a smooth workflow was a major challenge. Each part needs careful tuning to work well with the others. For instance, training the GAN to generate realistic 3D models requires significant computational resources and hyperparameter tuning.

Another challenge was maintaining a consistent state in the Streamlit app. Since the app reruns every time a button is clicked, I had to use session state to keep track of user inputs, model training, and optimization. (Still some bugs here!)

### Potential Improvements

- **Better Dataset**: Using a dataset like ShapeNet could improve the quality of the generated models.
- **Advanced Reinforcement Learning**: Switching from DQN to a more sophisticated method like Proximal Policy Optimization (PPO) could improve the optimization efficiency.
- **Enhanced Feedback Mechanism**: Adding a computer vision system to evaluate the structural integrity of generated designs could further enhance the quality and reliability of the output.
