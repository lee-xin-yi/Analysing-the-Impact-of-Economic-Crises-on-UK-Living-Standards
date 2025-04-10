import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import aic, bic
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import os

class GrangerCausalityAnalyzer:
    def __init__(self, data, maxlag=2, adf_significance=0.05, causality_significance=0.05):
        """
        Initializes the Granger Causality Analyzer.

        Parameters:
        - data: DataFrame containing the time-series variables.
        - maxlag: Maximum number of lags for Granger Causality tests (default=4).
        - adf_significance: Significance level for ADF test (default=0.05).
        - causality_significance: Significance level for Granger Causality test (default=0.05).

        Notes:
        - assumes time column labelled 'Date'
        """
        self.data = data
        self.maxlag = maxlag
        self.adf_significance = adf_significance
        self.causality_significance = causality_significance
        self.stationary_vars = {}
        self.stationarity_summary = {}
        self.causality_results = []

    def adf_test(self, series, name):
        """Performs the Augmented Dickey-Fuller (ADF) test for stationarity."""
        result = adfuller(series.dropna(), autolag='AIC')
        print(f'ADF Test for {name}:')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'Critical Values: {result[4]}')

        if result[1] <= self.adf_significance:
            print(f'{name} is stationary (reject null hypothesis)\n')
            return True, result[1]
        else:
            print(f'{name} is non-stationary (fail to reject null hypothesis)\n')
            return False, result[1]

    def transform_to_stationary(self):
        """Transforms non-stationary variables to stationary using differencing."""
        variables = [col for col in self.data.columns if col != 'Date']
        data_transformed = self.data[variables].copy()

        for var in variables:
            is_stationary, p_value = self.adf_test(data_transformed[var], var)
            if is_stationary:
                self.stationary_vars[var] = var
                self.stationarity_summary[var] = f"stationary (p-value = {p_value:.4f})"
            else:
                # First difference
                data_transformed[f'{var}_diff'] = data_transformed[var].diff()
                is_stationary, p_value = self.adf_test(data_transformed[f'{var}_diff'], f'Differenced {var}')
                if is_stationary:
                    self.stationary_vars[var] = f'{var}_diff'
                    self.stationarity_summary[var] = f"non-stationary, stationary after first differencing (p-value = {p_value:.4f})"
                else:
                    # Second difference
                    data_transformed[f'{var}_diff2'] = data_transformed[f'{var}_diff'].diff()
                    is_stationary, p_value = self.adf_test(data_transformed[f'{var}_diff2'], f'Second Differenced {var}')
                    if is_stationary:
                        self.stationary_vars[var] = f'{var}_diff2'
                        self.stationarity_summary[var] = f"non-stationary, stationary after second differencing (p-value = {p_value:.4f})"
                    else:
                        self.stationary_vars[var] = f'{var}_diff2'
                        self.stationarity_summary[var] = f"non-stationary even after second differencing (p-value = {p_value:.4f})"

        # Create final stationary dataset
        stationary_columns = [self.stationary_vars[var] for var in variables]
        data_stationary = data_transformed[stationary_columns].dropna()
        data_stationary.columns = variables  # Rename columns to original names for clarity
        self.data_stationary = data_stationary

        print("Stationary dataset created:")
        print(self.data_stationary.head())
        print("\nNaN values in stationary dataset:")
        print(self.data_stationary.isna().sum())

        return self.data_stationary

    def run_granger_causality_tests(self):
        """Runs Granger Causality tests on stationary variables."""
        print("\nRunning Granger Causality Tests...\n")
        variables = list(self.stationary_vars.keys())

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    print(f'Granger Causality Test: {var2} -> {var1}')
                    test_results = grangercausalitytests(self.data_stationary[[var1, var2]], maxlag=self.maxlag, verbose=True)

                    # Parse the results dictionary
                    for lag, result in test_results.items():
                        p_value = result[0]['ssr_ftest'][1]  # Extract p-value
                        if p_value <= self.causality_significance:
                            self.causality_results.append({'cause': var2, 'effect': var1, 'lag': lag, 'p_value': p_value})

                    print("\n" + "="*50 + "\n")

    def generate_conclusion(self):
        """Generates a summary of the findings from the Granger Causality tests."""
        print("\nConclusion: Granger Causality Analysis of Economic Indicators\n")

        # Stationarity Summary
        print("Stationarity Analysis:")
        for var, summary in self.stationarity_summary.items():
            print(f"- {var}: {summary}")
        print("\n")

        # Granger Causality Findings
        print("Granger Causality Findings:")
        if self.causality_results:
            for result in self.causality_results:
                print(f"- {result['cause']} Granger-causes {result['effect']} at lag {result['lag']} (p-value = {result['p_value']:.4f})")
        else:
            print("- No significant Granger Causality relationships were found.")
        print("\n")

    def run_workflow(self):
        """Executes the full workflow: transformation, Granger Causality, and conclusion."""
        self.transform_to_stationary()
        self.run_granger_causality_tests()
        self.generate_conclusion()


def plot_correlation_matrix(df: pd.DataFrame, title: str):
    """
    Plots a heatmap for the correlation matrix of all numeric columns (excluding 'Date') in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset containing the 'Date' column and numeric columns.
    - title (str): The title of the plot.
    """
    # Exclude the 'Date' column and select numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Calculate the correlation matrix for the numeric columns
    correlation_matrix = df[numeric_columns].corr()

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
    plt.title(title)
    plt.show()

def lasso_feature_selection(df, target_column, test_size=0.2, random_state=42):
    """
    Performs LASSO regression for feature selection on a dataset.

    Parameters:
    - df (pd.DataFrame): The dataset containing economic indicators and a target variable.
    - target_column (str): The name of the column to predict
    - test_size (float): The proportion of the dataset to use for testing.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - list: Selected features from LASSO.
    """
    # Drop target and non-numeric columns
    X = df.drop(columns=[target_column, 'Date'], errors='ignore')
    y = df[target_column]

    # Handle missing values by filling with the median
    X.fillna(X.median(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # Define hyperparameter range for alpha
    params = {"alpha": np.linspace(0.00001, 10, 500)}

    # Use K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Initialize LASSO and perform GridSearchCV
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X_scaled, y)

    # Get best alpha parameter
    best_alpha = lasso_cv.best_params_["alpha"]
    print(f"Best Alpha: {best_alpha}")

    # Train LASSO with best alpha
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X_train, y_train)

    # Select important features (non-zero coefficients)
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    print("Selected Features by LASSO:", selected_features)

    # Plot feature importance
    plt.figure(figsize=(15, 8))
    plt.bar(selected_features, lasso.coef_[lasso.coef_ != 0], color='blue')
    plt.xlabel("Features")
    plt.ylabel("LASSO Coefficient")
    plt.title("Feature Importance from LASSO Regression")
    plt.xticks(rotation=45)
    plt.show()

    return selected_features

class TimeSeriesRegression:
    def __init__(self, df, features, target, max_lag=6):
        """
        Initializes the TimeSeriesRegression model.

        Parameters:
        - df (pd.DataFrame): The dataset.
        - features (list): List of feature column names.
        - target (str): Target variable.
        - max_lag (int): Maximum lag to consider.
        """
        self.df = df.copy()
        self.features = features
        self.target = target
        self.max_lag = max_lag
        self.best_lags = {}

    def find_best_lags(self):
        """Finds the optimal lag for each feature using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)

        for feature in self.features:
            best_mse = float('inf')
            best_lag = 0

            for lag in range(1, self.max_lag + 1):
                df_lagged = self.df[[feature, self.target]].copy()
                df_lagged[f"{feature}_lag"] = df_lagged[feature].shift(lag)
                df_lagged.dropna(inplace=True)

                X = df_lagged[[f"{feature}_lag"]]
                y = df_lagged[self.target]

                mse_list = []

                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    mse_list.append(mse)

                avg_mse = np.mean(mse_list)

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_lag = lag

            self.best_lags[feature] = best_lag  # Store best lag

    def create_lagged_dataset(self):
        """Creates a DataFrame with the best lagged features."""
        self.df_lagged = self.df.copy()
        for feature, lag in self.best_lags.items():
            self.df_lagged[f"{feature}_lag"] = self.df_lagged[feature].shift(lag)

        self.df_lagged.dropna(inplace=True)  # Remove NaN rows
        return self.df_lagged

    def train_model(self):
        """Trains a multiple linear regression model using the best lags."""
        X = self.df_lagged[[f"{feat}_lag" for feat in self.best_lags.keys()]]
        y = self.df_lagged[self.target]

        self.model = LinearRegression()
        self.model.fit(X, y)

        # Predict & Evaluate
        self.y_pred = self.model.predict(X)
        self.mse = mean_squared_error(y, self.y_pred)
        print(f"Final Model MSE: {self.mse}")

    def plot_actual_vs_predicted(self):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_lagged[self.target], self.y_pred, color='blue', alpha=0.6, label="Actual vs Predicted")
        plt.plot(self.df_lagged[self.target], self.df_lagged[self.target], color='red', linestyle='--', label="Perfect Fit")
        plt.xlabel("Actual " + self.target)
        plt.ylabel("Predicted " + self.target)
        plt.title(f"Actual vs Predicted: {self.target}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals(self):
        """Plot residuals of the model."""
        residuals = self.df_lagged[self.target] - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_lagged[self.target], residuals, color='green', alpha=0.6, label="Residuals")
        plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
        plt.xlabel("Actual " + self.target)
        plt.ylabel("Residuals")
        plt.title(f"Residuals Plot: {self.target}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """Executes the full pipeline: find best lags, create dataset, train model, and plot results."""
        print("Finding best lags...")
        self.find_best_lags()
        print(f"Best lags: {self.best_lags}")

        print("Creating lagged dataset...")
        self.create_lagged_dataset()

        print("Training model...")
        self.train_model()

        # Plotting results
        self.plot_actual_vs_predicted()
        self.plot_residuals()
