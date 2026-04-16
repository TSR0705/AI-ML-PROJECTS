# Project Analysis: Prices Predictor System

## Overview
The **Prices Predictor System** is a production-grade machine learning project designed to predict housing prices. It leverages modern MLOps (Machine Learning Operations) practices to ensure that the entire lifecycle—from data ingestion to model deployment—is automated, reproducible, and scalable.

---

## 🏗 What is Made?
The project implements an end-to-end ML pipeline. It isn't just a script that trains a model; it's a complete system that includes:
- **Training Pipeline**: Automates data cleaning, feature engineering, model training, and evaluation.
- **Deployment Pipeline**: Implements **Continuous Deployment**, where a model is automatically deployed to a local server if it meets certain performance benchmarks.
- **Inference Pipeline**: A workflow for making predictions using the deployed model.
- **Exploratory Data Analysis (EDA)**: A detailed notebook documenting the data discovery phase.

---

## 🛠 Technologies and Tools Used
The project uses a sophisticated stack of industry-standard tools:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | **ZenML** | Manages the machine learning pipelines and ensures reproducibility. |
| **Tracking & Serving** | **MLflow** | Tracks experiments, logs metrics, and serves the model as a REST API. |
| **ML Framework** | **Scikit-learn** | Used for building regression models (Linear Regression, etc.). |
| **Data Processing** | **Pandas & Numpy** | Core libraries for data manipulation and mathematical operations. |
| **Design Patterns** | **Strategy Pattern** | Used in the `src` folder to make code modular and extensible. |
| **CLI & UI** | **Click & Rich** | Provides a professional command-line interface with styled output. |
| **Visualization** | **Matplotlib & Seaborn** | Used for generating insights during the EDA phase. |

---

## ⚙️ How it Works (The Workflow)

1. **Modular Source Code (`src/`)**: 
   The core logic is decoupled from the pipeline orchestration. Every step (e.g., handling missing values, training) is written as a "Strategy." For example, if you want to switch from Linear Regression to a Random Forest, you simply add a new strategy class in `src/model_building.py` without changing the pipeline logic.

2. **ZenML Steps (`steps/`)**: 
   These are thin wrappers around the `src/` logic. They integrate the code with ZenML's artifact tracking system.

3. **Pipelines (`pipelines/`)**:
   - **`training_pipeline`**: Ingests data, cleans it, engineers features, trains the model, and evaluates it.
   - **`deployment_pipeline`**: Takes the best-performing model and creates a local "Prediction Service" (a daemon process) that listens for inference requests.

4. **Experiment Tracking**:
   Every time you run the pipeline, MLflow captures the parameters, metrics (like MSE, R2 score), and the model itself. This allows you to compare different runs and rollback if needed.

---

## 📊 Evaluation of Code Quality

### 🌟 Strengths (High Quality)
- **Modularity**: The use of the **Strategy Design Pattern** is excellent. It follows the "Open/Closed Principle" (open for extension, closed for modification).
- **Type Safety**: Consistent use of Python **Type Hints** (`pd.DataFrame`, `RegressorMixin`, etc.) makes the code robust and easier to debug.
- **Documentation**: Methods have clear docstrings explaining parameters and return types.
- **Logging**: The project uses the `logging` module instead of `print` statements, which is a best practice for production systems.
- **Error Handling**: The code includes explicit checks (e.g., `isinstance(X_train, pd.DataFrame)`) to prevent runtime crashes.

### ⚠️ Areas for Improvement
- **Hardcoded Paths**: In `run_pipeline.py` and `training_pipeline.py`, some file paths are hardcoded to a specific user's directory (e.g., `/Users/ayushsingh/...`). This makes the code fail when run on a different machine without manual edits.
- **Configuration Management**: While there is a `config.yaml`, some parameters are still hardcoded in the function calls. Moving all hyperparameters to `config.yaml` would improve flexibility.

---

## Final Verdict
This project is an **excellent example of professional ML engineering**. It moves beyond the typical "Data Science Notebook" approach and builds a system that is ready for a production environment. It is well-structured, follows design patterns, and uses a powerful toolset (ZenML + MLflow).
