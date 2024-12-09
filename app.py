from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained logistic regression model and scaler
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data and convert to floats
        int_features = [float(x) for x in request.form.values()]
        final_features = np.array([int_features])  # Prepare input as array
        
        # Scale the input features
        final_features_scaled = scaler.transform(final_features)
        
        # Make prediction using the scaled data
        prediction = model.predict(final_features_scaled)
        
        # Interpret prediction result
        output = 'Approved' if prediction[0] == 1 else 'Not Approved'
        
        return render_template('index.html', prediction_text=f'Loan Status: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)


