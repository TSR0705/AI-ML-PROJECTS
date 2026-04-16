# House Price Prediction - Production ML Pipeline

## Problem
Predict house sale prices using the Ames Housing dataset with a production-ready, reproducible ML pipeline.

## Core Idea
End-to-end MLOps system using ZenML for orchestration and MLflow for experiment tracking. Not just a model—a complete training, deployment, and inference workflow with automated CD pipeline.

## Key Features
- **Modular architecture** using Strategy pattern for swappable components
- **Automated training pipeline** with data cleaning, feature engineering, and model evaluation
- **Continuous deployment** with MLflow model serving
- **Experiment tracking** with metrics, parameters, and model versioning
- **Production-ready inference** via REST API

---

## Architecture & Flow

```
Raw Data (ZIP/CSV) 
  → Data Ingestion 
  → Missing Values Handling 
  → Feature Engineering (log transform)
  → Outlier Detection (Z-score)
  → Train/Test Split
  → Model Training (Linear Regression)
  → Evaluation (MSE, RMSE, R²)
  → MLflow Tracking
  → Model Deployment (REST API)
  → Inference Service
```

### Components
- **`src/`**: Core ML logic with strategy pattern implementations
- **`steps/`**: ZenML pipeline steps (thin wrappers around src/)
- **`pipelines/`**: Training and deployment pipeline definitions
- **MLflow**: Experiment tracking + model serving
- **ZenML**: Pipeline orchestration + artifact management

### Interaction
1. ZenML orchestrates pipeline execution
2. Each step calls strategy classes from `src/`
3. MLflow logs metrics, parameters, and models
4. Deployment pipeline serves best model as REST API
5. Inference pipeline loads deployed model for predictions

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **ZenML** | Pipeline orchestration and reproducibility |
| **MLflow** | Experiment tracking and model serving |
| **scikit-learn** | ML algorithms (Linear Regression, Random Forest, etc.) |
| **pandas/numpy** | Data manipulation and numerical operations |
| **matplotlib/seaborn** | EDA visualizations |
| **click** | CLI interface |

---

## Folder Structure

```
├── src/                          # Core ML logic (strategy pattern)
│   ├── ingest_data.py           # Data loading (CSV, ZIP)
│   ├── handle_missing_values.py # Imputation strategies
│   ├── outlier_detection.py     # Z-score, IQR methods
│   ├── feature_engineering.py   # Log transform, scaling, encoding
│   ├── data_splitter.py         # Train/test split
│   ├── model_building.py        # Model training strategies
│   └── model_evaluator.py       # Metrics calculation
│
├── steps/                        # ZenML pipeline steps
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── outlier_detection_step.py
│   ├── feature_engineering_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── dynamic_importer.py      # Load batch data for inference
│   ├── prediction_service_loader.py
│   └── predictor.py             # Run predictions
│
├── pipelines/
│   ├── training_pipeline.py     # End-to-end training workflow
│   └── deployment_pipeline.py   # CD pipeline with model serving
│
├── analysis/
│   ├── EDA.ipynb                # Exploratory data analysis
│   └── analyze_src/             # EDA modules (univariate, bivariate, etc.)
│
├── data/
│   └── archive.zip              # Raw dataset
│
├── extracted_data/
│   └── AmesHousing.csv          # Extracted dataset
│
├── mlruns/                       # MLflow experiment artifacts
├── mlflow.db                     # MLflow tracking database
│
├── config.yaml                   # Pipeline configuration
├── requirements.txt              # Python dependencies
├── run_pipeline.py               # Execute training pipeline
├── run_deployment.py             # Execute deployment pipeline
└── sample_predict.py             # Test inference API
```

---

## Setup & Run

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### Run Training Pipeline
```bash
python run_pipeline.py
```

### View Experiments
```bash
# Get MLflow tracking URI from pipeline output, then:
mlflow ui --backend-store-uri <tracking_uri>
# Open http://localhost:5000
```

### Deploy Model
```bash
python run_deployment.py
```

### Test Inference
```bash
# Model serves at http://127.0.0.1:8000/invocations
python sample_predict.py
```

### Common Issues
- **Import errors**: Ensure you're in project root when running scripts
- **ZenML not initialized**: Run `zenml init` first
- **Port conflicts**: MLflow default port 5000 may conflict; use `--port` flag
- **Model not found**: Run training pipeline before deployment

---

## Screenshots

### MLflow Experiment Tracking
![MLflow Experiments](images/Screenshot%202026-04-16%20171808.png)
*Experiment runs with tracked metrics (MSE, RMSE, R²) and parameters*

### Model Metrics Comparison
![Model Metrics](images/Screenshot%202026-04-16%20171904.png)
*Performance comparison across different runs*

### ZenML Pipeline Execution
![Pipeline Run](images/Screenshot%202026-04-16%20171923.png)
*Visual representation of pipeline steps and artifacts*

---

## API Usage

### Prediction Endpoint
```bash
POST http://127.0.0.1:8000/invocations
Content-Type: application/json
```

### Request Format
```json
{
  "dataframe_records": [
    {
      "Order": 1,
      "PID": 5286,
      "MS SubClass": 20,
      "Lot Frontage": 80.0,
      "Lot Area": 9600,
      "Overall Qual": 5,
      "Gr Liv Area": 1710.0,
      ...
    }
  ]
}
```

### Response
```json
{
  "predictions": [175000.0]
}
```

---

## Configuration

Edit `config.yaml` to customize:
```yaml
training_pipeline:
  data_path: "data/archive.zip"
  target_column: "SalePrice"
  feature_engineering:
    strategy: "log"  # log, standard, minmax
    features: ["Gr Liv Area", "SalePrice"]
  outlier_detection:
    column_name: "SalePrice"
```

---

## Limitations & Improvements

### Current Limitations
- **Hardcoded paths**: Some absolute paths in code (needs environment variables)
- **Single model type**: Only Linear Regression in main pipeline (Random Forest, XGBoost available but not integrated)
- **No hyperparameter tuning**: Fixed parameters, no grid search
- **Local deployment only**: No cloud deployment (AWS, GCP, Azure)
- **No data validation**: Missing schema validation and data drift detection

### Practical Improvements
1. **Add hyperparameter optimization** (Optuna, GridSearchCV)
2. **Implement model comparison** (auto-select best model)
3. **Add data validation** (Great Expectations, Evidently)
4. **Cloud deployment** (AWS SageMaker, GCP Vertex AI)
5. **API authentication** (JWT tokens, API keys)
6. **Monitoring dashboard** (Grafana, Prometheus)
7. **CI/CD integration** (GitHub Actions, Jenkins)
8. **Feature store** (Feast, Tecton)

---

## Design Patterns Used

- **Strategy Pattern**: Swappable algorithms in `src/` modules
- **Factory Pattern**: Create strategy instances dynamically
- **Pipeline Pattern**: Sequential data transformations

See `explanations/` folder for detailed examples.

---

## License
Apache 2.0
