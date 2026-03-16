import torch
import numpy as np
import joblib
from model import LSTMModel

def load_artifacts():

    model = LSTMModel(input_size=14, hidden_size=32, num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load("artifacts/best_model.pth", map_location="cpu"))
    model.eval()

    imputer = joblib.load("artifacts/imputer.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    y_scaler = joblib.load("artifacts/y_scaler.pkl")

    return model, imputer, scaler, y_scaler

def predict(input_sequence):

    model, imputer, scaler, y_scaler = load_artifacts()

    input_sequence = np.array(input_sequence)

    # Flatten for preprocessing
    flat = input_sequence.reshape(-1, input_sequence.shape[-1])
    flat = imputer.transform(flat)
    flat = scaler.transform(flat)

    processed = flat.reshape(1, 30, 14)

    tensor_input = torch.tensor(processed, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(tensor_input).numpy()

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1,1))

    return float(pred.flatten()[0])

if __name__ == "__main__":

    # Example dummy input (30 timesteps, 14 features)
    sample_input = np.random.rand(30, 14)

    prediction = predict(sample_input)
    print("Prediction:", prediction)
