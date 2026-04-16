# AI-ML-PROJECTS

Production-grade Machine Learning projects demonstrating end-to-end ML workflows, from exploratory analysis to deployment.

---

## Projects

### [PROJECT-01: Real Estate Price Prediction](PROJECT-01%20ML%20FOR%20ESTATE%20PRICE%20PRIDICTION/)

Classic ML regression project predicting Boston housing prices with model comparison and feature engineering.

**Key Features:**
- Stratified train-test split with feature engineering (TAXRM ratio)
- Model comparison: Linear Regression, Decision Tree, Random Forest
- Cross-validation with RMSE evaluation
- Trained model persistence (joblib)

**Best Model:** Random Forest (Test RMSE: ~2.98)

**Tech Stack:** Python, scikit-learn, pandas, numpy, matplotlib

**Quick Start:**
```bash
cd "PROJECT-01 ML FOR ESTATE PRICE PRIDICTION"
python dragon_real_estate_predictor_clean.py
```

[View Full Documentation →](PROJECT-01%20ML%20FOR%20ESTATE%20PRICE%20PRIDICTION/README.md)

---

### [PROJECT-02: House Price Prediction - Production ML Pipeline](PROJECT-02-%20LINEAR%20REGRESSION%20PRICES%20PRIDICTION%20WITH%20FULL%20PIPELINE/)

Enterprise-grade MLOps system with ZenML orchestration, MLflow tracking, and automated deployment.

**Key Features:**
- End-to-end pipeline: ingestion → cleaning → feature engineering → training → deployment
- Strategy pattern architecture for swappable components
- Automated continuous deployment with model serving
- Experiment tracking with MLflow (metrics, parameters, artifacts)
- REST API for real-time predictions

**Architecture:** ZenML pipelines + MLflow serving + Strategy pattern

**Tech Stack:** ZenML, MLflow, scikit-learn, pandas, numpy, click

**Quick Start:**
```bash
cd "PROJECT-02- LINEAR REGRESSION PRICES PRIDICTION WITH FULL PIPELINE"
pip install -r requirements.txt
zenml init
python run_pipeline.py
```

**Deploy & Serve:**
```bash
python run_deployment.py
# Model serves at http://127.0.0.1:8000/invocations
python sample_predict.py
```

[View Full Documentation →](PROJECT-02-%20LINEAR%20REGRESSION%20PRICES%20PRIDICTION%20WITH%20FULL%20PIPELINE/README.md)

---

## Project Comparison

| Feature | PROJECT-01 | PROJECT-02 |
|---------|-----------|-----------|
| **Focus** | Model comparison | Production pipeline |
| **Orchestration** | Manual scripts | ZenML automated |
| **Tracking** | None | MLflow experiments |
| **Deployment** | Saved model file | REST API serving |
| **Architecture** | Monolithic | Modular (Strategy pattern) |
| **Reproducibility** | Limited | Full pipeline versioning |
| **Use Case** | Learning/prototyping | Production-ready |

---

## Getting Started

### Prerequisites
```bash
# PROJECT-01
pip install pandas numpy scikit-learn matplotlib jupyter joblib

# PROJECT-02
pip install -r "PROJECT-02- LINEAR REGRESSION PRICES PRIDICTION WITH FULL PIPELINE/requirements.txt"
zenml init
```

### Repository Structure
```
AI-ML-PROJECTS/
├── PROJECT-01 ML FOR ESTATE PRICE PRIDICTION/
│   ├── dragon_real_estate_predictor_clean.py
│   ├── Dragon.joblib (trained model)
│   ├── house_data.csv
│   └── README.md
│
└── PROJECT-02- LINEAR REGRESSION PRICES PRIDICTION WITH FULL PIPELINE/
    ├── src/              # Core ML logic (strategy pattern)
    ├── steps/            # ZenML pipeline steps
    ├── pipelines/        # Training & deployment pipelines
    ├── config.yaml       # Pipeline configuration
    ├── run_pipeline.py   # Execute training
    └── README.md
```

---

## Learning Path

1. **Start with PROJECT-01** - Understand basic ML workflow and model comparison
2. **Move to PROJECT-02** - Learn production MLOps with orchestration and deployment

---

## Author
[TSR0705](https://github.com/TSR0705)

## License
Open source - Educational and commercial use permitted
