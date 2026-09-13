import base64
import os
from io import BytesIO

import joblib
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'usa_brown_coal_simplified_all.csv')


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        app.logger.error(f"Error loading model: {exc}")
        return None


model = load_model()


def load_historical_data():
    df = pd.read_csv(DATA_PATH)
    return df[['year', 'quantity']].dropna().reset_index(drop=True)


def build_chart_data_url(prediction_year=None, prediction_value=None):
    data = load_historical_data()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(
        data['year'],
        data['quantity'],
        color='#60a5fa',
        linewidth=2.5,
        label='Historical consumption'
    )

    if prediction_year is not None and prediction_value is not None:
        ax.scatter([prediction_year], [prediction_value], color='#f59e0b', s=90, zorder=5)
        ax.annotate(
            f'{int(prediction_year)}: {prediction_value:,}',
            (prediction_year, prediction_value),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            color='#fcd34d'
        )
        ax.plot([prediction_year, prediction_year], [0, prediction_value], linestyle='--', color='#f59e0b', alpha=0.5)

    ax.set_title('USA Brown Coal Consumption Trend')
    ax.set_xlabel('Year')
    ax.set_ylabel('Thousand metric tons')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4)
    ax.legend(loc='upper left')

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150)
    plt.close(fig)

    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', chart_data_url=build_chart_data_url())


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

        chart_data_url = build_chart_data_url(year_value, output)
        return render_template(
            'index.html',
            prediction_text=(
                f'Predicted USA brown coal consumption for {int(year_value):,}: '
                f'{output:,} thousand metric tons'
            ),
            chart_data_url=chart_data_url
        )
    except Exception:
        return render_template(
            'index.html',
            error_message='Invalid input. Please enter a valid year, such as 2025.',
            chart_data_url=build_chart_data_url()
        )


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)