# Notebooks

This folder contains the analysis and modeling notebook for the **House Prices - Advanced Regression Techniques** (Kaggle) task.

## Notebook list

| Notebook | Description |
|---|---|
| [`House-Price-Prediction-Advanced-Regression-Techniques.ipynb`](./House-Price-Prediction-Advanced-Regression-Techniques.ipynb) | EDA, data preprocessing, and training regression models (Linear, Ridge, Lasso) on the house price dataset |

Open directly in Google Colab: [Open in Google Colab](https://colab.research.google.com/github/aio25-mix002/m05-p0501/blob/notebooks/House-Price-Prediction-Advanced-Regression-Techniques.ipynb)

## Input data

The notebook reads data from:
```
../data/train-house-prices-advanced-regression-techniques.csv
```
- **1,460 rows x 81 columns**, target column is `SalePrice`.
- If the data file is not present, download it with the command from the repo's root README:
  ```bash
  gdown 1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd -O data/train-house-prices-advanced-regression-techniques.csv
  ```

## Notebook structure

1. **Load Dataset** — read the CSV, inspect `head()`, `shape`, `describe()`.
2. **EDA (Exploratory Data Analysis)**
   - `SalePrice` distribution (histogram + mean line).
   - Bar chart of missing value counts per column.
   - Correlation heatmap of numerical features.
   - Boxplots for the 8 numeric features most correlated with `SalePrice` (`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `YearBuilt`).
3. **Preprocessing**
   - Drop columns with >50% missing values: `Id`, `Alley`, `PoolQC`, `Fence`, `MiscFeature`.
   - Train/test split, 75/25 (`random_state=42`) → train: **1,095 rows**, test: **365 rows**.
   - Split into 36 numeric columns and 39 categorical columns.
   - Fill missing values: `"none"` for categorical columns, `SimpleImputer` (mean) for numeric columns.
   - One-hot encode categorical columns (`OneHotEncoder`).
4. **Normalization**
   - Scale numeric columns with `MinMaxScaler`.
   - Concatenate scaled numeric columns with one-hot encoded columns → final feature matrix: **282 columns**.
5. **Training Regression Models**
   - Train 3 models: `LinearRegression`, `Ridge`, `Lasso` on the 282-feature set.
6. **Advanced Techniques — Polynomial Regression**
   - Generate degree-2 polynomial features (`PolynomialFeatures(degree=2, interaction_only=True)`) from the numeric columns → **666 features**, re-scale with `MinMaxScaler`, concatenate with the one-hot columns → **912 columns** total.
   - Re-train the same 3 models on this expanded feature set.

## Results

### Table 1 — Linear / Ridge / Lasso on the base feature set (282 features)

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---:|---:|---:|---:|
| **Lasso** | 20,032.67 | **26,362.31** | 0.9339 | **0.9008** |
| LinReg | 20,029.22 | 28,137.00 | 0.9339 | 0.8870 |
| Ridge | 22,488.29 | 28,587.34 | 0.9167 | 0.8833 |

### Table 2 — Linear / Ridge / Lasso on the polynomial feature set (912 features)

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---:|---:|---:|---:|
| **Lasso** | 20,032.67 | **26,362.31** | 0.9339 | **0.9008** |
| LinReg | 20,029.22 | 28,137.00 | 0.9339 | 0.8870 |
| Ridge | 22,488.29 | 28,587.34 | 0.9167 | 0.8833 |

**Observations:**
- **Lasso** performs best on both train and test (lowest Test RMSE, highest Test R² at ~0.90), with a smaller train/test gap than Ridge and LinReg — i.e. less overfitting, thanks to L1-based feature selection.
- LinearRegression and Lasso have nearly identical Train RMSE, but Lasso generalizes better on the test set.
- ⚠️ **Note (likely bug):** in the Polynomial Regression training cell, the models are still fit on `X_train`/`X_test` (the original 282-column feature set) instead of `X_train_poly`/`X_test_poly` (the 912-column polynomial feature set). As a result, Table 2 is currently **identical** to Table 1 and does not actually reflect the effect of the polynomial features. The `model().fit(...)` call should be updated to use `X_train_poly`/`X_test_poly`, then the notebook re-run to get real results.

## How to run

```bash
# Set up the environment (see the repo's root README)
uv venv
uv sync

# Open the notebook
jupyter notebook notebooks/House-Price-Prediction-Advanced-Regression-Techniques.ipynb
```

Or open it directly via the Google Colab link above.
