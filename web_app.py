from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load models
with open('encoder', 'rb') as f:
    oh = pickle.load(f)

with open('scaler', 'rb') as f:
    sc = pickle.load(f)

with open('model', 'rb') as f:
    reg = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        rd = float(request.form['rd'])
        ad = float(request.form['ad'])
        mar = float(request.form['mar'])
        state = request.form['state']

        d1 = {
            'R&D Spend': [rd],
            'Administration': [ad],
            'Marketing Spend': [mar],
            'State': [state]
        }

        newdf = pd.DataFrame(d1)

        # One Hot Encoding
        newdf[oh.get_feature_names_out()] = oh.transform(newdf[['State']])
        newdf.drop(columns=oh.feature_names_in_, inplace=True)

        # Scaling
        newdf[['R&D Spend', 'Administration', 'Marketing Spend']] = sc.transform(
            newdf[['R&D Spend', 'Administration', 'Marketing Spend']]
        )

        # Prediction
        result = reg.predict(newdf)
        final = result[0]

        return render_template('index.html',
                               prediction_text=f'Predicted Profit is {final:.2f} $')

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)