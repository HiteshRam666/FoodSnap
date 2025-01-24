import gradio as gr
import os
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv, find_dotenv 

# Set up OpenAI API key
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Setup class names
with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Model and transforms preparation ###    
# Create model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101,
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="EffnetB2_feature_extracted_model.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. LangChain Setup for Nutritional Information and Recipes ###

# Create the LangChain prompt templates
nutrition_prompt_template = """
Provide a brief description and nutritional information about {food_name}. 
Include calories, macronutrients (proteins, carbs, fats), and any interesting facts in highlighted bullet points.
"""

recipe_prompt_template = """
Create a simple and easy-to-follow recipe for {food_name}. 
Include the list of ingredients and step-by-step instructions.
"""

nutrition_prompt = PromptTemplate(input_variables=["food_name"], template=nutrition_prompt_template)
recipe_prompt = PromptTemplate(input_variables=["food_name"], template=recipe_prompt_template)

# Initialize OpenAI model with LangChain
llmModel = OpenAI(max_tokens=1000)

# Create chains for nutritional information and recipes
nutrition_chain = nutrition_prompt | llmModel
recipe_chain = recipe_prompt | llmModel

def fetch_nutritional_info(food_name: str) -> str:
    """Fetch nutritional information for the given food name using LangChain."""
    try:
        # Run the LangChain chain to get the response
        response = nutrition_chain.invoke({"food_name": food_name})
        return response.strip()
    except Exception as e:
        return f"Could not fetch nutritional information. Error: {str(e)}"

def fetch_recipe(food_name: str) -> str:
    """Fetch recipe suggestions for the given food name using LangChain."""
    try:
        # Run the LangChain chain to get the response
        response = recipe_chain.invoke({"food_name": food_name})
        return response.strip()
    except Exception as e:
        return f"Could not fetch recipe. Error: {str(e)}"

### 4. Predict function ###

def predict(img) -> Tuple[Dict, float, str, str]:
    """Transforms and performs a prediction on img and returns prediction, time taken, nutritional info, and recipe."""
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Get the top prediction
    top_prediction = max(pred_labels_and_probs, key=pred_labels_and_probs.get)
    
    # Fetch nutritional information and recipe for the top prediction
    nutritional_info = fetch_nutritional_info(top_prediction)
    recipe = fetch_recipe(top_prediction)
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary, prediction time, nutritional information, and recipe
    return pred_labels_and_probs, pred_time, nutritional_info, recipe

### 5. Gradio app ###

# Create title, description, and article strings
title = "FoodSnap😋🍕📸🤳"
description = "FoodSnap is an advanced image classification application designed to identify and classify images of food into 101 different categories. Powered by a pretrained EfficientNetB2 model fine-tuned on the Food101 dataset, the app offers fast and accurate predictions with an intuitive interface. Additionally, it provides detailed nutritional information and easy-to-follow recipes for the top-predicted food item using OpenAI's language model integration via LangChain."
article = ""

# Create examples images
example_list = [['1.jpg'],
                ['2.jpg'],
                ['3.jpg'],
                ['4.jpg'],
                ['5.jpg']]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
        gr.Textbox(label="Nutritional Information"),
        gr.Textbox(label="Recipe Suggestion"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the app!
demo.launch()
