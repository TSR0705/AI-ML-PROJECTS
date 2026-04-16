import logging
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from src.ingest_data import DataIngestorFactory
from src.handle_missing_values import MissingValueHandler, FillMissingValuesStrategy
from src.feature_engineering import FeatureEngineer, LogTransformation, MinMaxScaling, OneHotEncoding, StandardScaling
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run():
    config = get_config()
    tp_config = config.get("training_pipeline", {})
    
    data_path = tp_config.get("data_path", "data/archive.zip")
    target_column = tp_config.get("target_column", "SalePrice")
    fe_strategy = tp_config.get("feature_engineering", {}).get("strategy", "log")
    features = tp_config.get("feature_engineering", {}).get("features", ["Gr Liv Area", "SalePrice"])

    logging.info("1. Ingestion Step")
    ingestor = DataIngestorFactory.get_data_ingestor(".zip")
    raw_data = ingestor.ingest(data_path)
    
    logging.info("2. Handling Missing Values Step")
    mv_handler = MissingValueHandler(FillMissingValuesStrategy(method="mean"))
    filled_data = mv_handler.handle_missing_values(raw_data)
    
    logging.info("3. Feature Engineering Step")
    if fe_strategy == "log":
        fe_strat_obj = LogTransformation(features)
    elif fe_strategy == "standard_scaling":
        fe_strat_obj = StandardScaling(features)
    elif fe_strategy == "minmax_scaling":
        fe_strat_obj = MinMaxScaling(features)
    elif fe_strategy == "onehot_encoding":
        fe_strat_obj = OneHotEncoding(features)
    else:
        raise ValueError("Unsupported strategy")
        
    engineer = FeatureEngineer(fe_strat_obj)
    engineered_data = engineer.apply_feature_engineering(filled_data)
    
    logging.info("4. Outlier Detection Step")
    # For simplicity, passing only numeric data to the outlier detector
    df_numeric = engineered_data.select_dtypes(include=["int64", "float64"])
    # Need to keep the target column if it was removed in previous steps or is not numeric, 
    # but here SalePrice is numeric.
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    # It modifies the DataFrame and handles outliers
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    
    logging.info("5. Data Splitter Step")
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df_cleaned, target_column)

    logging.info("6. Model Building Step")
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
    
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        
        logging.info("Building and training model...")
        pipeline.fit(X_train, y_train)

        logging.info("7. Evaluator Step")
        X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)
        evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
        metrics = evaluator.evaluate(pipeline.named_steps["model"], X_test_processed, y_test)
        
        logging.info(f"Evaluation Metrics: {metrics}")
        for k, v in metrics.items():
            mlflow.log_metric(k.replace(" ", "_"), v)

    print("\nPipeline Execution Complete!")
    print(f"Run `mlflow ui` to see the results.")

if __name__ == "__main__":
    run()
