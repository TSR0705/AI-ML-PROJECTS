# PROJECT-01: ML for Real Estate Price Prediction

## Overview
This project implements a machine learning model to predict real estate prices using the Boston Housing dataset. It demonstrates the complete ML pipeline from data exploration to model deployment.

## Dataset
The Boston Housing dataset contains 506 samples with 14 features:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's (Target Variable)

## Files

- `dragon_real_estate_predictor_clean.py` - Standalone Python script (recommended)
- `dragon_real_estate_predictor.py` - Converted from Jupyter notebook
- `Dragon real state.ipynb` - Original Jupyter notebook with analysis
- `Dragon.joblib` - Trained Random Forest model
- `house_data.csv` - Main dataset
- `housing.data` - Raw data file
- `housing.names` - Feature descriptions
- `OUTPUT OF DIFF MODELS.txt` - Model comparison results
- `TESTING.ipynb` - Model testing notebook

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib jupyter joblib
```

## Usage

### Running the Standalone Script
```bash
cd "PROJECT-01 ML FOR ESTATE PRICE PRIDICTION"
python dragon_real_estate_predictor_clean.py
```

### Running the Jupyter Notebook
```bash
jupyter notebook "Dragon real state.ipynb"
```

### Loading the Trained Model
```python
from joblib import load
model = load('Dragon.joblib')

# Make predictions
predictions = model.predict(your_data)
```

## Model Performance

The project compares three regression models:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor** (Best Performance)

### Random Forest Results:
- **Training RMSE**: ~1.31
- **Cross-Validation RMSE**: ~3.40 (±0.74)
- **Test RMSE**: ~2.98

## Pipeline

The ML pipeline includes:
1. **Data Loading & Exploration**
2. **Train-Test Split** (Stratified by CHAS feature)
3. **Feature Engineering** (TAXRM = TAX/RM ratio)
4. **Data Preprocessing**:
   - Missing value imputation (median strategy)
   - Feature scaling (StandardScaler)
5. **Model Training** (Random Forest)
6. **Cross-Validation** (10-fold)
7. **Model Evaluation** (RMSE, MSE)
8. **Model Persistence** (joblib)

## Key Findings

- **RM** (average number of rooms) has the strongest positive correlation with price (0.695)
- **LSTAT** (% lower status population) has the strongest negative correlation (-0.738)
- The engineered feature **TAXRM** shows moderate negative correlation (-0.538)
- Random Forest outperforms Linear Regression and Decision Tree models

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Feature selection techniques
- Ensemble methods (XGBoost, LightGBM)
- Deep learning approaches
- Additional feature engineering

## Author
TSR0705

## License
Open source - Educational purposes
