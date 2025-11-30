Restaurant Rating Prediction (Zomato Dataset)
This project is a Flask web application that predicts restaurant ratings using a machine learning model trained on the Zomato Bangalore restaurants dataset.​
It also provides analysis pages such as a location heatmap, model performance metrics, and top recommendation examples.​

Features
Predicts rating for a single restaurant using:
Location, restaurant name, cuisines, restaurant type, approximate cost, votes, online order, table booking.
Colored result card with explanation (green / yellow / red band).
Compare Mode: side‑by‑side prediction for two restaurants.

Separate analysis pages:
Location Heatmap (average ratings by area).
Model Performance (R², MAE, RMSE).
Top Recommendation example tables by area.

Project structure
train_model.py – data cleaning, feature encoding and model training script.
src/app/app.py – Flask application (routes for prediction, compare, heatmap, metrics, top pages).
src/app/templates/
index.html – main prediction UI.
compare.html – compare mode.
heatmap.html – location heatmap page.
metrics.html – model performance page.
top.html – top recommendation examples.
The dataset file (zomato.csv) and trained model file (model.pkl) are not included in the repo because they are large; instructions to regenerate them are below.​

How to run locally
Clone the repository
bash
git clone https://github.com/Mugeshbuba/zomato_prediction.git
cd zomato_prediction

Create and activate virtual environment
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

Install dependencies
bash
pip install -r requirements.txt
(If requirements.txt is not present, install manually: flask, pandas, numpy, scikit-learn etc.)​

Download dataset
Download the Zomato Bangalore restaurants dataset from Kaggle (or other source) and save it as:
text
zomato.csv
in the project root.

Train the model
bash
python train_model.py

This script will:
Clean and preprocess the dataset.
Train the rating prediction model.
Save the trained model as src/app/model.pkl.
Save the encoded label objects and the location_avg.csv used for the heatmap.

Run the Flask app
bash
cd src/app
python app.py
Open the URL shown in the terminal (usually http://127.0.0.1:5000/) in a browser.

Usage
On the home page, enter restaurant details and click Predict Rating.
Use Compare Mode (button on top‑right) to open the comparison page in a new tab.
Use the left menu (☰) to open:
Heatmap (new tab).
Model Performance (new tab).
Top Recommendations (new tab).
Notes
Dataset and model files are ignored via .gitignore due to GitHub’s 100 MB file size limit; regenerate them using train_model.py as described.​

The “Top Recommendations” page currently uses example restaurants; it can be extended to automatically query the dataset and show dynamic top‑N recommendations per area, similar to recommendation systems described in related work.​

You can customize text, add screenshots of your UI, and link your report or paper if needed.
