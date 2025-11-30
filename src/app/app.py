from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# Load model and encoders (paths relative to this file)
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, 'model.pkl'))
le_location = joblib.load(os.path.join(base_dir, 'le_location.pkl'))
le_cuisines = joblib.load(os.path.join(base_dir, 'le_cuisines.pkl'))
le_rest_type = joblib.load(os.path.join(base_dir, 'le_rest_type.pkl'))

@app.route('/')
def home():
    # Performance metrics (use your real numbers)
    metrics = {"r2": 0.82, "mae": 0.28, "rmse": 0.42}

    # Load location averages for heatmap cards
    heatmap_path = os.path.join(base_dir, 'location_avg.csv')
    location_avgs = []
    if os.path.exists(heatmap_path):
        df_loc = pd.read_csv(heatmap_path)
        # take first 6–8 locations to display
        location_avgs = df_loc.sort_values('avg_rating', ascending=False).head(6).to_dict(orient='records')
    return render_template('index.html',metrics=metrics,location_avgs=location_avgs)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Restaurant name (for display only)
        name = request.form.get('name', '').strip()

        location = request.form['location']
        cuisines = request.form['cuisines']
        rest_type = request.form['rest_type']
        approx_cost = float(request.form['approx_cost'])
        votes = int(request.form['votes'])
        online_order = 1 if request.form['online_order'] == 'Yes' else 0
        book_table = 1 if request.form['book_table'] == 'Yes' else 0

        # Encode categories
        loc_encoded = le_location.transform([location])[0]
        cuisines_encoded = le_cuisines.transform([cuisines])[0]
        rest_type_encoded = le_rest_type.transform([rest_type])[0]

        # Build input dataframe
        input_df = pd.DataFrame([{
            'location': loc_encoded,
            'cuisines': cuisines_encoded,
            'rest_type': rest_type_encoded,
            'approx_cost(for two people)': approx_cost,
            'votes': votes,
            'online_order': online_order,
            'book_table': book_table
        }])

        # Simple explanation text
        explanation_parts = []
        if votes > 500:
            explanation_parts.append(
                "This restaurant has many votes, so the rating is based on a lot of customer feedback."
            )
        else:
            explanation_parts.append(
                "This restaurant has fewer votes, so the rating is based on limited feedback."
            )
        if approx_cost > 800:
            explanation_parts.append("Higher cost usually indicates a more premium experience.")
        else:
            explanation_parts.append("Lower cost usually indicates a more budget‑friendly place.")
        explanation_text = " ".join(explanation_parts)

        # Predict rating
        pred = model.predict(input_df)[0]
        rating_value = round(pred, 2)

        # Card color based on rating
        if rating_value >= 4.0:
            color = "green"
        elif rating_value >= 3.5:
            color = "yellow"
        else:
            color = "red"

        return jsonify({
            'name': name,
            'rating': rating_value,
            'explanation': explanation_text,
            'color': color
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/compare')
def compare():
    # Separate page for compare mode
    return render_template('compare.html')

@app.route('/heatmap')
def heatmap():
    heatmap_path = os.path.join(base_dir, 'location_avg.csv')
    location_avgs = []
    if os.path.exists(heatmap_path):
        df_loc = pd.read_csv(heatmap_path)
        location_avgs = df_loc.sort_values('avg_rating', ascending=False).to_dict(orient='records')
    return render_template('heatmap.html', location_avgs=location_avgs)

@app.route('/metrics')
def metrics_page():
    metrics = {"r2": 0.82, "mae": 0.28, "rmse": 0.42}  # your real values
    return render_template('metrics.html', metrics=metrics)

@app.route('/top')
def top_page():
    return render_template('top.html')

if __name__ == '__main__':
    app.run(debug=True)
