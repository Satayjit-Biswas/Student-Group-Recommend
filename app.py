from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained machine learning model
try:
    with open('model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading the model: {e}")
    rf_model = None

# Group mapping for clarity
group_mapping = {0: "Arts", 1: "Commerce", 2: "Science"}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Overview page
    return render_template('index.html')

@app.route('/recommend')
def recommend_ui():
    # Input form page
    return render_template('recommend.html')

@app.route('/predict_group', methods=['POST'])
def predict_group():
    # Get user input from the form
    try:
        age = int(request.form['age'])
        location = int(request.form['location'])  # Encoded values: 0, 1, 2, 3
        family_size = int(request.form['family_size'])
        studytime = int(request.form['studytime'])
        attendance = float(request.form['attendance'])
        english = float(request.form['english'])
        math = float(request.form['math'])
        science = float(request.form['science'])
        social_science = float(request.form['social_science'])
        art_culture = float(request.form['art_culture'])

        # Calculate total_score and average_score
        total_score = english + math + science + social_science + art_culture
        average_score = total_score / 5

        # Create user input feature list
        user_input = [
            age, location, family_size, studytime, attendance,
            english, math, science, social_science, art_culture,
            total_score, average_score
        ]

        # Predict group and probabilities
        if rf_model:
            predicted_group = rf_model.predict([user_input])[0]
            probabilities = rf_model.predict_proba([user_input])[0]

            # Map group prediction and probabilities to human-readable form
            predicted_group_label = group_mapping.get(predicted_group, "Unknown")
            probability_dict = {
                "Arts": round(probabilities[0] * 100, 2),
                "Commerce": round(probabilities[1] * 100, 2),
                "Science": round(probabilities[2] * 100, 2)
            }

            return render_template(
                'predict_group.html',
                group=predicted_group_label,
                probabilities=probability_dict
            )
        else:
            return render_template('predict_group.html', group="Model unavailable", probabilities={})

    except ValueError:
        # Handle invalid input gracefully
        return render_template('predict_group.html', group="Invalid Input", probabilities={})

if __name__ == '__main__':
    app.run(debug=True)
