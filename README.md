# Retail Demand Forecasting Pipeline

End-to-end **Deep Learning + MLOps system** for retail demand forecasting using **PyTorch, PostgreSQL, MLflow, FastAPI, Docker, CI/CD, and AWS EC2**.

## Project Overview ->

This project implements a **production-ready machine learning pipeline** for forecasting retail demand from multivariate time-series data.

It covers the **complete ML lifecycle**:

- Data Engineering  
- Feature Engineering (SQL)  
- Model Training (Deep Learning + Classical ML)  
- Experiment Tracking  
- API Deployment  
- Containerization  
- Cloud Deployment  
- CI/CD Automation

## System Architecture ->

- PostgreSQL (Feature Engineering)
- Training Pipeline (PyTorch LSTM / GRU)
- MLflow (Experiment Tracking)
- Saved Model Artifacts (model, scaler, imputer)
- FastAPI Inference Service
- Docker Container
- AWS EC2 Deployment
- CI/CD Pipeline (GitHub Actions)
- User Prediction Requests

## Model Development ->

### A. Deep Learning Models
- Implemented **LSTM and GRU networks in PyTorch**
- Built **custom training loops** with:
  - Batch processing
  - Gradient descent optimization
  - Validation monitoring

### B. Feature Engineering (SQL)

Performed feature engineering using PostgreSQL:

- Lag Features:
  - `lag_1`, `lag_7`, `lag_30`
- Rolling Statistics:
  - `rolling_7`, `rolling_30`

Captured:
- Seasonality  
- Short-term demand patterns  
- Temporal dependencies  

### C. Classical Machine Learning

- Trained **Random Forest Regressor** on engineered features  
- Compared performance with deep learning models  

## Model Performance ->

| Model | MSE |
|-------|-----|
| LSTM (PyTorch) | 2266 |
| Random Forest | 897 |

Achieved **~60% improvement** using feature-engineered tabular model over LSTM baseline.


## Key Insights ->

- Deep learning models (LSTM/GRU) were effective but **underperformed compared to tree-based models**  
- Explicit feature engineering captured temporal patterns more effectively  
- Demonstrates importance of **model selection based on data characteristics**  

## MLOps & Productionization ->

### A. Experiment Tracking (MLflow)

- Logged:
  - Hyperparameters  
  - Metrics  
  - Model artifacts  
- Ensured reproducibility of experiments  

### B. Inference API (FastAPI)

- Built REST API for real-time predictions  
- Accepts time-series input and returns forecast  

### C. Docker Containerization
- Containerized application using Docker
- Ensures:
    - Environment consistency
    - Portability
    - Easy deployment

### D. AWS Deployment (EC2)
- Deployed container on Amazon EC2 (Ubuntu)
- Exposed public API endpoint

### E. CI/CD Pipeline (GitHub Actions)
- Automated:
    - Docker build
    - Deployment to EC2
- Triggered on every push to main branch

## How to Run Locally ->
1. Build Docker Image
- docker build -t retail-api . <br>

2. Run Container
- docker run -p 8000:8000 retail-api <br>

3. Access API
- http://localhost:8000/docs

## Deployment Steps (AWS EC2) ->
1. Launch EC2 instance (Ubuntu 22.04) <br>
2. Install Docker <br>
3. Clone repository <br>
4. Build Docker image <br>
5. Run container 

## Future Improvements ->
- Adding Transformer-based time-series model
- Implementing monitoring & logging system
- Integrating model registry (MLflow / S3)
- Adding authentication to API
- Using load balancing (Nginx / AWS ALB)

## Skills Demonstrated ->
- Deep Learning (LSTM, GRU, PyTorch, Tensorflow, Keras)
- Feature Engineering (SQL, Time-Series)
- Classical ML (Random Forest)
- Experiment Tracking (MLflow)
- API Development (FastAPI)
- Containerization (Docker)
- Cloud Deployment (AWS EC2)
- CI/CD Automation (GitHub Actions)

## Summary ->
Built a production-grade ML system that goes beyond model training by incorporating:
- Engineering best practices
- Deployment pipelines
- Real-world system design

## Author ->
**Ayush Saxena**
GitHub: https://github.com/DRIFT-619
