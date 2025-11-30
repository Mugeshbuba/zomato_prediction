# 🍽️ Restaurant Rating Prediction

ML + Flask app to predict Zomato restaurant ratings and show helpful insights.  

## ✨ Features

- 🔮 Predict restaurant rating from user inputs  
- ⚖️ Compare two restaurants side‑by‑side  
- 🗺️ Location heatmap with average area ratings  
- 📊 Model performance page (R², MAE, RMSE)  
- ⭐ Top recommendation examples by area  

## 🚀 How to run

1. Clone the repo and go inside:  
   `git clone https://github.com/Mugeshbuba/zomato_prediction.git`  
2. Create venv and install requirements.  
3. Add `zomato.csv` dataset (local only) and run `train_model.py` to create `model.pkl`.  
4. Run `python app.py` from `src/app` and open `http://127.0.0.1:5000/`.  
