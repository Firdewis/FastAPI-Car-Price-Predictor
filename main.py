from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load the trained preprocessor and model
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("best_rf_model_new.pkl", "rb") as file:
    model = pickle.load(file)

# Define the FastAPI application
app = FastAPI()

# Define feature columns (excluding Price and ID)
numerical_cols = ['Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Levy']
categorical_cols = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 
                    'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Airbags']

# Define the input data model (Pydantic BaseModel)
class CarInput(BaseModel):
    Prod_year: float  # Production year
    Engine_volume: float  # Engine volume
    Mileage: float  # Mileage (numeric, removing 'km')
    Cylinders: float  # Number of cylinders
    Doors: float  # Number of doors (numeric, removing '04-May' like values)
    Levy: float  # Tax fee
    Manufacturer: str  # Manufacturer
    Model: str  # Model name
    Category: str  # Category
    Leather_interior: str  # Leather interior (Yes/No)
    Fuel_type: str  # Fuel type
    Gear_box_type: str  # Gearbox type
    Drive_wheels: str  # Drive wheels type
    Wheel: str  # Steering wheel position
    Color: str  # Color of the car
    Airbags: str  # Number of airbags

# Define the output data model
class CarOutput(BaseModel):
    predicted_price: float  # Predicted price of the car

@app.post("/infos")
def send_back_information(input: CarInput):
    return {"message": input.dict()}

@app.post("/predict", response_model=CarOutput)
def predict_car_price(input: CarInput):
    try:
        # Convert input data into a dictionary
        input_dict = input.dict()
        
        # Convert into DataFrame for preprocessing
        features_df = pd.DataFrame([{
            'Prod. year': input_dict['Prod_year'],
            'Engine volume': input_dict['Engine_volume'],
            'Mileage': input_dict['Mileage'],
            'Cylinders': input_dict['Cylinders'],
            'Doors': input_dict['Doors'],
            'Levy': input_dict['Levy'],
            'Manufacturer': input_dict['Manufacturer'],
            'Model': input_dict['Model'],
            'Category': input_dict['Category'],
            'Leather interior': input_dict['Leather_interior'],
            'Fuel type': input_dict['Fuel_type'],
            'Gear box type': input_dict['Gear_box_type'],
            'Drive wheels': input_dict['Drive_wheels'],
            'Wheel': input_dict['Wheel'],
            'Color': input_dict['Color'],
            'Airbags': input_dict['Airbags']
        }])

        # Apply the full preprocessor (including numeric imputation and categorical encoding)
        processed_features = preprocessor.transform(features_df)

        # Make a prediction using the trained model
        prediction = model.predict(processed_features)[0]

        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
