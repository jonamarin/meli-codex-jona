from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# Load the model and preprocessing pipeline
with open('xgb_model.sav', 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    label_encoder = model_data['label_encoder']

app = FastAPI()

class Item(BaseModel):
    sub_status: str
    deal_ids: str
    seller_id: int
    listing_type_id: str
    price: float
    buying_mode: str
    tags: str
    category_id: str
    last_updated: str
    international_delivery_mode: str
    id: str
    accepts_mercadopago: int
    currency_id: str
    thumbnail: str
    title: str
    automatic_relist: int
    date_created: str
    secure_thumbnail: str
    stop_time: int
    status: str
    initial_quantity: int
    start_time: int
    permalink: str
    sold_quantity: int
    available_quantity: int
    seller_country: str
    seller_state: str
    seller_city: str
    free_shipping: int
    local_pick_up: int
    days_active: int
    num_payment_methods: int
    payment_type_N: int
    payment_type_D: int
    payment_type_C: int
    payment_type_G: int

@app.post("/predict")
async def predict(item: Item):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([item.dict()])
        
        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)
        prediction_label = label_encoder.inverse_transform(prediction)[0]
        
        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
