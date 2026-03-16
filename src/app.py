import torch
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Defining model architecture
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

# Loading trained model
model = LSTMModel()
model.load_state_dict(torch.load("artifacts/best_model.pth", map_location=torch.device("cpu")))
model.eval()
scaler_y = joblib.load("artifacts/y_scaler.pkl")

# Creating FastAPI app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    data: List[List[float]]  # expects 30 x 14

# Print message
@app.get("/")
def home():
    return {"message": "Retail Demand Forecasting API is running"}

# Prediction function
@app.post("/predict")
def predict(input_data: InputData):

    input_array = np.array(input_data.data)

    # Validate shape
    if input_array.shape != (30, 14):
        return {"error": "Input must be of shape (30, 14)"}

    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)

        prediction_np = prediction.detach().numpy().reshape(-1, 1)

        prediction_real = scaler_y.inverse_transform(prediction_np)

    return {"prediction": float(prediction_real.item())}
