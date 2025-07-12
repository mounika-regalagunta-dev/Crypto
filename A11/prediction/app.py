from flask import Flask, render_template, request
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models
models = {
    "Bitcoin": joblib.load('models/bitcoin/linear_regression_model.pkl'),
    "Ethereum": joblib.load('models/ethereum/gradient_boosting_model.pkl'),
    "Binance": joblib.load('models/binance/linear_regression_model.pkl'),
    "Dogecoin": joblib.load('models/dogecoin/linear_regression_model.pkl'),
    "Cosmos": joblib.load('models/cosmos/linear_regression_model.pkl'),
    "Litecoin": joblib.load('models/litecoin/linear_regression_model.pkl'),
    "Stellar": joblib.load('models/stellar/linear_regression_model.pkl'),
    "Ripple": joblib.load('models/ripple/linear_regression_model.pkl'),
    "Cardano": joblib.load('models/cardano/linear_regression_model.pkl'),
    "Solana": joblib.load('models/solana/linear_regression_model.pkl')
}

# Load scalers (Only for coins that require them)
scalers = {
    "Bitcoin": joblib.load('scalers/bitcoin/scaler.pkl'),
    "Ethereum": joblib.load('scalers/ethereum/scaler.pkl'),
    "Binance": joblib.load('scalers/binance/scaler.pkl'),
    "Litecoin": joblib.load('scalers/litecoin/scaler.pkl'),
    "Solana": joblib.load('scalers/solana/scaler.pkl')
}

# Load last 100 data points
last_100_data = {
    "Bitcoin": np.load('last_100/bitcoin/last_100.npy'),
    "Ethereum": np.load('last_100/ethereum/last_100.npy'),
    "Binance": np.load('last_100/binance/last_100.npy'),
    "Dogecoin": np.load('last_100/dogecoin/last_100.npy'),
    "Cosmos": np.load('last_100/cosmos/last_100.npy'),
    "Litecoin": np.load('last_100/litecoin/last_100.npy'),
    "Stellar": np.load('last_100/stellar/last_100.npy'),
    "Ripple": np.load('last_100/ripple/last_100.npy'),
    "Cardano": np.load('last_100/cardano/last_100.npy'),
    "Solana": np.load('last_100/solana/last_100.npy')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        coin = request.form.get('coin')
        num_days = int(request.form.get('num_days', 0))

        if not coin or coin not in models:
            return render_template('index.html', error="Invalid coin selection.")

        # Get the appropriate model, scaler, and last_100 data
        model = models[coin]
        scaler = scalers.get(coin, None)  # Some coins do not require a scaler
        last_100 = last_100_data[coin].copy()  # Avoid modifying the original data

        # Generate future predictions
        future_predictions = []
        dates = []
        current_date = datetime.now()

        for _ in range(num_days):
            next_day = model.predict(last_100)

            # If scaler exists, inverse transform the prediction
            prediction = scaler.inverse_transform(next_day.reshape(1, -1))[0, 0] if scaler else next_day[0, 0]

            future_predictions.append(round(prediction, 2))
            dates.append((current_date + timedelta(days=len(future_predictions))).strftime("%Y-%m-%d"))

            # Update last_100 with the new predicted value
            last_100 = np.append(last_100[:, 1:], next_day.reshape(1, 1), axis=1)

        # Generate the Matplotlib graph
        plt.figure(figsize=(8, 4))
        plt.plot(dates, future_predictions, marker='o', linestyle='-', color='b', label=f"{coin} Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{coin} Price Prediction")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()  # Close to free memory

        return render_template('index.html', predictions=zip(dates, future_predictions), graph_url=graph_url)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__== '__main__':
    app.run(debug=True)