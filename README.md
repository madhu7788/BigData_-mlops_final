# MLOps Time-Series Regression Project

## Overview
This project demonstrates an end-to-end MLOps workflow using a time-series regression dataset.

## Dataset
- Public Bike Sharing Dataset
- Time-based split (35% train, 35% validation, 30% test)

## Modeling
- H2O AutoML used for model discovery
- Manual training of:
  - Random Forest
  - XGBoost
  - Gradient Boosting (Champion)

## MLOps Stack
- MLflow Tracking Server (AWS EC2)
- Backend Store: Neon PostgreSQL
- Artifact Store: AWS S3
- Model Registry used for versioning

## Deployment
- FastAPI used for serving predictions
- Swagger UI enabled

## Drift Analysis
- Performed using NannyML on newer data window

## How to Run
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
