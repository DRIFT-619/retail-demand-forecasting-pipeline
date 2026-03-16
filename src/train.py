import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model import LSTMModel
import joblib

def main():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("LSTM_TimeSeries_Forecasting")
    print("Tracking URI:", mlflow.get_tracking_uri())

    # Loading data
    X = np.load("C:\\Users\\reign\\OneDrive\\Desktop\\All Folders\\1. University Modules Folder\\1. Projects\\retail-demand-forecasting-pipeline\\data\\final\\X.npy", allow_pickle=True)
    y = np.load("C:\\Users\\reign\\OneDrive\\Desktop\\All Folders\\1. University Modules Folder\\1. Projects\\retail-demand-forecasting-pipeline\\data\\final\\y.npy", allow_pickle=True)
    dates_array = np.load("C:\\Users\\reign\\OneDrive\\Desktop\\All Folders\\1. University Modules Folder\\1. Projects\\retail-demand-forecasting-pipeline\\data\\final\\dates_array.npy", allow_pickle=True)
    
    # Splitting data ->

    # Defining split dates
    train_split_date = np.datetime64("2015-07-01")
    val_split_date   = np.datetime64("2015-07-03")

    # Masks
    train_idx = dates_array < train_split_date
    val_idx   = (dates_array >= train_split_date) & (dates_array < val_split_date)
    test_idx  = dates_array >= val_split_date

    # Splitting data
    X_train_seq = X[train_idx]
    y_train_seq = y[train_idx]

    X_val_seq = X[val_idx]
    y_val_seq = y[val_idx]

    X_test_seq = X[test_idx]
    y_test_seq = y[test_idx]

    # Imputing Null Values ->
    X_train_flat = X_train_seq.reshape(-1, 14)
    X_val_flat = X_val_seq.reshape(-1, 14)
    X_test_flat = X_test_seq.reshape(-1, 14)

    imputer = SimpleImputer(strategy="mean")

    X_train_flat = imputer.fit_transform(X_train_flat)
    X_val_flat = imputer.transform(X_val_flat)
    X_test_flat = imputer.transform(X_test_flat)

    X_train_seq = X_train_flat.reshape(-1, 30, 14)
    X_val_seq = X_val_flat.reshape(-1, 30, 14)
    X_test_seq = X_test_flat.reshape(-1, 30, 14)

    # Scaling Features
    scaler = StandardScaler()

    # Flattening time dimension to scale features
    X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_val_flat   = X_val_seq.reshape(-1, X_val_seq.shape[-1])
    X_test_flat  = X_test_seq.reshape(-1, X_test_seq.shape[-1])

    # Fitting only on train data
    scaler.fit(X_train_flat)

    # Transforming and reshaping back to 3D
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train_seq.shape)
    X_val_scaled   = scaler.transform(X_val_flat).reshape(X_val_seq.shape)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(X_test_seq.shape)

    # Scaling the Target variable
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(y_train_seq.reshape(-1,1)).flatten()
    y_val_scaled = y_scaler.transform(y_val_seq.reshape(-1,1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_seq.reshape(-1,1)).flatten()

    # Model Training ->

    # Creating Custom Dataset Class
    class SalesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
        

    train_dataset = SalesDataset(X_train_scaled, y_train_scaled)
    val_dataset = SalesDataset(X_val_scaled, y_val_scaled)
    test_dataset = SalesDataset(X_test_scaled, y_test_scaled)

    # Creating DataLoader
    BATCH_SIZE = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Wrapping Training in MLflow Run
    with mlflow.start_run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialising Model
        INPUT_SIZE = X_train_scaled.shape[2]
        HIDDEN_SIZE = 32
        NUM_LAYERS = 1
        DROPOUT = 0.0

        model = LSTMModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        # Defining Loss & Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Creating a Manual Training Loop
        EPOCHS = 10
        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            
            # Training
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(X_batch)
                
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = model(X_batch)
                    
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)

            # No Early Stopping implemented here
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(model.state_dict(), "best_lstm.pth")

            # Code for Early Stopping
            '''
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered")
                break
            '''
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f}")
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Loading the saved Model
        # model.load_state_dict(torch.load("best_lstm.pth", weights_only=True))

        # Making Predictions 
        model.eval()

        preds = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                
                preds.append(outputs.cpu().numpy())
                actuals.append(y_batch.numpy())

        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)

        preds = y_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        actuals = y_scaler.inverse_transform(actuals.reshape(-1,1)).flatten()

        # Computing RMSE
        rmse_lstm = np.sqrt(mean_squared_error(actuals, preds))
        print("Test RMSE:", rmse_lstm)

        # MLflow Logging ->

        # Log hyperparameters
        mlflow.log_param("model_type", "PyTorch_LSTM")
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("sequence_length", 30)
        mlflow.log_param("num_features", INPUT_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        # Log metrics
        mlflow.log_metric("test_rmse", rmse_lstm)

        # Log model
        mlflow.pytorch.log_model(model, name="model")

        torch.save(model.state_dict(), "artifacts/best_model.pth")
        
        mlflow.log_artifact("artifacts/best_model.pth")

        print("Run logged successfully.")

        # Logging Scaler Artifact

        joblib.dump(imputer, "artifacts/imputer.pkl")
        joblib.dump(scaler, "artifacts/scaler.pkl")
        joblib.dump(y_scaler, "artifacts/y_scaler.pkl")

        mlflow.log_artifact("artifacts/imputer.pkl")
        mlflow.log_artifact("artifacts/scaler.pkl")
        mlflow.log_artifact("artifacts/y_scaler.pkl")

if __name__ == "__main__":
    main()
