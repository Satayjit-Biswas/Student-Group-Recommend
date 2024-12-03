from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pickled machine learning model (e.g., RandomForest)
try:
    with open('model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)
except (pickle.UnpicklingError, TypeError):
    print("Error occurred while loading the model.")
    rf_model = None  # Placeholder in case model loading fails

# Dictionary to map numeric group labels to actual group names
group_mapping = {0: "Arts", 1: "Commerce", 2: "Science"}

app = Flask(__name__)

@app.route('/')
def index():
    # Just show an overview on this page
    return render_template('index.html')

@app.route('/recommend')
def recommend_ui():
    # Show the form to get user input on this page
    return render_template('recommend.html')

@app.route('/predict_group', methods=['POST'])
def predict_group():
    # Get user input from the form
    english = float(request.form['english'])
    math = float(request.form['math'])
    science = float(request.form['science'])
    social_science = float(request.form['social_science'])
    art_culture = float(request.form['art_culture'])
    
    # Calculate total_score and average_score
    total_score = english + math + science + social_science + art_culture
    average_score = total_score / 5

    # Get other input data
    user_input = [
        int(request.form['age']),
        int(request.form['location']),  # 0 = Rural, 1 = Urban
        int(request.form['family_size']),
        int(request.form['studytime']),
        float(request.form['attendance']),
        english,
        math,
        science,
        social_science,
        art_culture,
        total_score,  # Add total_score as a feature
        average_score  # Add average_score as a feature
    ]
    
    # Ensure model is loaded before using it for prediction
    if rf_model:
        # Make prediction
        predicted_group = rf_model.predict([user_input])
        predicted_group_label = group_mapping.get(predicted_group[0], "Unknown")  # Map the numeric result to label
        
        # Render a separate page to display the result
        return render_template('predict_group.html', stu_group=predicted_group_label)
    else:
        return render_template('predict_group.html', stu_group="Model is not available for prediction at the moment.")

if __name__ == '__main__':
    app.run(debug=True)
