import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        app.logger.error(f"Error loading model: {exc}")
        return None


model = load_model()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template(
            'index.html',
            error_message='Prediction service is currently unavailable. The ML model could not be loaded.'
        )

    year = request.form.get('year', '').strip()

    try:
        year_value = float(year)
        if year_value <= 0:
            raise ValueError('Year must be a positive number.')

        prediction = model.predict(np.array([[year_value]]))
        output = int(round(float(prediction[0])))

        return render_template(
            'index.html',
            prediction_text=(
                f'Predicted USA brown coal consumption for {int(year_value):,}: '
                f'{output:,} thousand metric tons'
            )
        )
    except Exception:
        return render_template(
            'index.html',
            error_message='Invalid input. Please enter a valid year, such as 2025.'
        )


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)