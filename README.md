```markdown
# FastAPI-Car-Price-Predictor 🚗💰
## Car Price Prediction API

This project is a **FastAPI-based API** for predicting car prices using a **machine learning model** trained on real-world car price data. It uses a **RandomForestRegressor** with preprocessing for both numerical and categorical features.

---

## 🔍 Project Overview
- **📊 Dataset:** [sidharth178/car-prices-dataset](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset) (train.csv, test.csv)
- **🧠 Model:** `RandomForestRegressor` with feature engineering (imputation + one-hot encoding)
- **🚀 API:** Built with **FastAPI** for real-time car price prediction
- **📌 Features:** 16 features including `Prod. year`, `Engine volume`, `Mileage`, etc.
- **🎯 Output:** **Predicted car price** (numeric value)

---

## 📦 Requirements
- **Python 3.12**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## ⚙️ How to Run the API
1. **Clone this repository:**
   ```bash
   git clone https://github.com/Firdewis/FastAPI-Car-Price-Predictor.git
   cd FastAPI-Car-Price-Predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate  # Windows
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Test the API in Swagger UI:**
   - Open in your browser: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🔥 API Endpoints
### **1️⃣ Health Check**
- **`GET /`** → Returns a welcome message.

### **2️⃣ Predict Car Price**
- **`POST /predict`**
- **Input:** JSON object with 16 car features.
- **Output:** Predicted price (`float`).

**Example Request:**
```json
{
  "Prod_year": 2015,
  "Engine_volume": 2.5,
  "Mileage": 100000,
  "Cylinders": 4,
  "Doors": 4,
  "Levy": 500,
  "Manufacturer": "Toyota",
  "Model": "Corolla",
  "Category": "Sedan",
  "Leather_interior": "Yes",
  "Fuel_type": "Petrol",
  "Gear_box_type": "Automatic",
  "Drive_wheels": "Front",
  "Wheel": "Left",
  "Color": "White",
  "Airbags": "6"
}
```

**Example Response:**
```json
{
  "predicted_price": 12500.50
}
```

---

