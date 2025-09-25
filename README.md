# Regression-Models

# Regression Models for Macroeconomic Forecasting

> “History never repeats itself, but it does often rhyme.” — Mark Twain

**A comprehensive Python framework for supervised machine learning applied to macroeconomic forecasting, with focus on predicting industrial production using diverse regression techniques.**

This project explores advanced regression methodologies for forecasting the Industrial Production Index (INDPRO) using a rich set of macroeconomic indicators. By comparing traditional subset selection with modern machine learning approaches including penalized regression, ensemble methods, and tree-based models, the framework demonstrates how to balance model complexity with predictive accuracy in economic forecasting applications.

## Overview

Economic forecasting presents unique challenges: high-dimensional datasets, multicollinearity among predictors, structural breaks, and the need for interpretable models. This project addresses these challenges through systematic comparison of regression techniques, from classical forward selection to modern ensemble methods, all applied to predicting monthly changes in U.S. industrial production.

## Features

### Core Methodologies
- **Forward Subset Selection**: Stepwise variable selection using information criteria
- **Penalized Regression**: Ridge and Lasso regularization for high-dimensional data
- **Partial Least Squares**: Dimensionality reduction with supervised learning
- **Tree-Based Models**: Decision trees with ensemble improvements
- **Ensemble Learning**: Random Forest and Gradient Boosting implementations

### Data Processing
- **FRED-MD Integration**: Direct access to monthly macroeconomic dataset
- **Time Series Transformations**: Automatic stationarity adjustments
- **Lag Structure**: Multi-period predictor relationships
- **Train/Test Splitting**: Proper temporal validation methodology

### Model Evaluation
- **Cross-Validation**: K-fold and Leave-One-Out validation
- **Information Criteria**: AIC/BIC for model selection
- **Feature Importance**: Variable significance ranking
- **Performance Metrics**: RMSE comparison across methods

## Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels tqdm
```

### Required Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation and time series analysis
- `matplotlib` - Visualization and plotting
- `scikit-learn` - Machine learning algorithms
- `statsmodels` - Statistical modeling and regression
- `tqdm` - Progress tracking for model fitting
- `finds` - Custom economic data library (included)

### API Setup
```python
# Configure FRED API access in secret.py
credentials = {
    'fred': {
        'api_key': 'your_fred_api_key'  # Get free key from FRED
    }
}
```

## Quick Start

### Data Loading and Preparation

```python
import numpy as np
import pandas as pd
from finds.readers.alfred import Alfred, fred_md

# Initialize FRED data access
alf = Alfred(api_key=credentials['fred']['api_key'])

# Load FRED-MD dataset
df, transforms = fred_md()
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

### Basic Forecasting Setup

```python
# Define target variable and prediction horizon
target_id = 'INDPRO'  # Industrial Production Index
lags = 3              # Include up to 3 lags of predictors

# Create lagged feature matrix
Y = data[target_id].iloc[lags:]
X = pd.concat([
    data.shift(lag).iloc[lags:].rename(
        columns=lambda col: f"{col}_{lag}"
    ) for lag in range(1, lags+1)
], axis=1)

print(f"Target series: {len(Y)} observations")
print(f"Feature matrix: {X.shape}")
```

### Model Training Example

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Split data temporally
split_date = 20221231
X_train = X[Y.index <= split_date]
X_test = X[Y.index > split_date]
Y_train = Y[Y.index <= split_date]
Y_test = Y[Y.index > split_date]

# Standardize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Lasso regression with cross-validation
model = LassoCV(cv=5, random_state=42).fit(X_train_scaled, Y_train)
predictions = model.predict(X_test_scaled)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print(f"Test RMSE: {rmse:.6f}")
```

## Project Structure

```
regression-models/
├── README.md
├── requirements.txt
├── secret.py                    # API credentials (not in repo)
├── notebooks/
│   └── macroeconomic_forecasting.ipynb
├── finds/
│   ├── readers/
│   │   └── alfred.py           # FRED data access
│   └── utils.py                # Plotting utilities
├── models/
│   ├── subset_selection.py     # Forward selection implementation
│   ├── penalized_regression.py # Ridge/Lasso models
│   ├── ensemble_methods.py     # Tree-based models
│   └── evaluation.py           # Model comparison utilities
├── data/
│   └── fred_md/               # Cached datasets
└── results/
    ├── model_comparison.png    # Performance charts
    └── feature_importance/     # Variable importance plots
```

## Methodologies

### Forward Subset Selection

Sequential variable selection using information criteria:

```python
def forward_select(Y, X, selected, ic='bic'):
    """Forward selection using AIC/BIC criteria"""
    remaining = [x for x in X.columns if x not in selected]
    results = []
    for x in remaining:
        model = sm.OLS(Y, X[selected + [x]]).fit()
        results.append({
            'variable': x,
            'aic': model.aic,
            'bic': model.bic,
            'rsquared_adj': model.rsquared_adj
        })
    return pd.DataFrame(results).sort_values(by=ic).iloc[0]
```

**Key Features:**
- Information criteria optimization (AIC/BIC)
- Automatic stopping based on penalty
- Interpretable variable selection
- Computational efficiency

### Penalized Regression

Regularization techniques for high-dimensional data:

**Ridge Regression (L2 Penalty):**
- Shrinks coefficients toward zero
- Handles multicollinearity effectively  
- Retains all variables with reduced impact

**Lasso Regression (L1 Penalty):**
- Performs automatic variable selection
- Sets some coefficients exactly to zero
- Creates sparse, interpretable models

### Ensemble Methods

**Random Forest:**
- Bootstrap aggregation of decision trees
- Feature randomness at each split
- Robust to overfitting
- Built-in feature importance

**Gradient Boosting:**
- Sequential error correction
- Adaptive learning from residuals
- High predictive accuracy
- Tunable complexity

## Key Results

### Dataset Coverage
- **Time Period**: 1964-2024 (726 monthly observations)
- **Variables**: 122 macroeconomic indicators (after missing data removal)
- **Training Period**: 1964-2022
- **Test Period**: 2023-2024

### Model Performance Comparison

| Method | Parameters | Train RMSE | Test RMSE | Key Features |
|--------|------------|------------|-----------|--------------|
| Forward Selection | k=29 | 0.005955 | 0.007674 | Interpretable, BIC-optimal |
| Lasso Regression | α=0.000255 | 0.006279 | 0.006808 | Sparse, 82 variables |
| Ridge Regression | α=1.8 | 0.004305 | 0.009825 | All variables, regularized |
| Random Forest | depth=15 | 0.003760 | 0.005875 | Non-linear, robust |
| Gradient Boosting | depth=3 | 0.002704 | 0.007407 | Sequential learning |
| PLS Regression | k=11 | 0.005306 | 0.008791 | Dimension reduction |

### Top Predictive Variables

**Most Important Predictors:**
1. **CLAIMS_1**: Initial Claims (1-month lag)
2. **BUSLOANS_1**: Commercial & Industrial Loans
3. **M2SL_1**: M2 Money Supply
4. **USGOOD_1**: Goods-Producing Employment
5. **UNRATE_1**: Unemployment Rate

## Usage Examples

### Cross-Validation Model Selection

```python
from sklearn.model_selection import cross_val_score, KFold

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compare models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(
        model, X_train_scaled, Y_train, 
        cv=kf, scoring='neg_mean_squared_error'
    )
    results[name] = np.sqrt(-scores.mean())
    
pd.Series(results).plot(kind='bar', title='Cross-Validation RMSE')
```

### Feature Importance Analysis

```python
# Extract feature importance from tree-based models
def plot_feature_importance(model, feature_names, top_n=15):
    importance = pd.Series(
        model.feature_importances_, 
        index=feature_names
    ).sort_values(ascending=False)[:top_n]
    
    importance.plot(kind='barh', title=f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return importance
```

### Time Series Visualization

```python
def plot_predictions(y_true, y_pred, title="Forecast Comparison"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative changes for better visualization
    pd.concat([
        y_true.cumsum(), 
        pd.Series(y_pred, index=y_true.index).cumsum()
    ], axis=1).plot()
    
    plt.title(title)
    plt.legend(['Actual', 'Predicted'])
    plt.ylabel('Cumulative Change')
    plt.tight_layout()
```

## Research Applications

### Academic Research
- **Macroeconomic Forecasting**: Comparative methodology studies
- **Variable Selection**: Feature importance in economic prediction
- **Model Evaluation**: Out-of-sample performance assessment

### Policy Applications
- **Economic Monitoring**: Real-time industrial production nowcasting
- **Recession Prediction**: Early warning system development
- **Policy Impact**: Monetary and fiscal policy effect modeling

### Business Intelligence
- **Supply Chain Planning**: Industrial capacity forecasting
- **Investment Strategy**: Sector rotation based on production trends
- **Risk Management**: Economic cycle timing models

## Technical Notes

### Data Transformations
FRED-MD applies standardized transformations to ensure stationarity:
- **Level**: Raw data (transformation code 1)
- **First Difference**: Δx_t (transformation code 2)  
- **Log Level**: ln(x_t) (transformation code 4)
- **Log Difference**: Δln(x_t) (transformation code 5)
- **Growth Rates**: Various percentage change calculations

### Time Series Considerations
- **Temporal Splitting**: Train/test split respects time order
- **Lag Structure**: Prevents look-ahead bias
- **Missing Values**: Forward-fill recent observations
- **Structural Breaks**: Model evaluation across different economic regimes

### Computational Performance
- **Parallel Processing**: Multi-core cross-validation
- **Memory Efficiency**: Sparse matrix operations where applicable
- **Scalability**: Handles 300+ potential predictors

## Data Sources & Attribution

- **FRED-MD Dataset**: Federal Reserve Bank of St. Louis
- **Industrial Production**: Federal Reserve Board
- **Employment Data**: Bureau of Labor Statistics
- **Financial Variables**: Treasury Department, Federal Reserve

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Implement new regression method with proper evaluation
4. Add tests and documentation
5. Submit pull request with performance benchmarks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

These models are for research and educational purposes. Economic forecasts should not be used as the sole basis for policy or investment decisions. Model performance may vary significantly during structural breaks or economic crises. Always validate results with domain experts and consider multiple modeling approaches.

*"Supervised machine learning and regression models for macroeconomic forecasting aim to strike a balance between model complexity and accuracy in predicting economic indicators."*
