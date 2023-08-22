import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import openpyxl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the updated data
updated_df = pd.read_excel('Book1.xlsx')

# Exclude weekends
updated_df = updated_df[updated_df['Date'].dt.weekday < 5]

# Use the data from January to July for modeling
train_data_updated = updated_df['TotalCases']

# Define and fit the SARIMA model using the updated training data
sarima_model_updated = SARIMAX(train_data_updated, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
sarima_fit_updated = sarima_model_updated.fit(disp=False)

# Calculate the date range for the next 20 weekdays
last_date = updated_df['Date'].max()
next_weekdays = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=20, freq='B')

# Make predictions for the next 20 weekdays
forecast_updated = sarima_fit_updated.get_forecast(steps=len(next_weekdays))
forecast_values_updated = forecast_updated.predicted_mean
confidence_intervals_updated = forecast_updated.conf_int()

# Update the forecast dataframe
forecast_df_updated_corrected = pd.DataFrame({
    'Date': next_weekdays,
    'Forecast': forecast_values_updated,
    'Lower CI': confidence_intervals_updated.iloc[:, 0],
    'Upper CI': confidence_intervals_updated.iloc[:, 1]
})

# Export the corrected forecast dataframe to an Excel file
forecast_file_path_updated = "forecasted_data_next_20_weekdays.xlsx"
forecast_df_updated_corrected.to_excel(forecast_file_path_updated, index=False)

# Calculate residuals for the updated model
residuals_updated = sarima_fit_updated.resid

# Re-perform the Ljung-Box test on the residuals
lb_test_updated = sm.stats.acorr_ljungbox(residuals_updated, lags=[10], return_df=True)

# Create a DataFrame for validation data
validation_data = {
    'AIC': [sarima_fit_updated.aic],
    'BIC': [sarima_fit_updated.bic],
    'Ljung-Box Statistic': [lb_test_updated['lb_stat'].values[0]],
    'Ljung-Box P-Value': [lb_test_updated['lb_pvalue'].values[0]]
}
validation_df = pd.DataFrame(validation_data)

# Export the validation data to an Excel file
validation_file_path = "validation_data.xlsx"
validation_df.to_excel(validation_file_path, index=False)

# Hypothesis and Null Hypothesis
hypothesis = "H0: The residuals are independently distributed (no autocorrelation)."
null_hypothesis = "H1: The residuals have significant autocorrelation."

# Print the hypothesis and null hypothesis
print("Hypothesis: ", hypothesis)
print("Null Hypothesis: ", null_hypothesis)

# Insert an empty line
print("")

# Validation message
validation_message = "Model validation: "
if lb_test_updated['lb_pvalue'].values[0] < 0.05:
    validation_message += "Not validated (high autocorrelation in residuals)."
else:
    validation_message += "Validated (no significant autocorrelation in residuals)."

# Print the validation message
print(validation_message)

# Valid hypothesis message
if lb_test_updated['lb_pvalue'].values[0] < 0.05:
    valid_hypothesis = "The null hypothesis is rejected. Autocorrelation is present in the residuals."
else:
    valid_hypothesis = "The null hypothesis is valid. No significant autocorrelation in the residuals."

# Insert an empty line
print("")

# Print the valid hypothesis message
print(valid_hypothesis)


# Create the plots for residuals, Ljung-Box test, and model information criteria
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot the residuals
axes[0].plot(updated_df['Date'][:len(residuals_updated)], residuals_updated, color='green')
axes[0].set_title('Residuals Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residuals')
axes[0].grid(True)

# ACF plot of the residuals (for Ljung-Box test visualization)
sm.graphics.tsa.plot_acf(residuals_updated, lags=30, ax=axes[1])
axes[1].set_title('Autocorrelation Function (ACF) of Residuals (Ljung-Box Test)')

# AIC and BIC values as a bar plot
aic_bic_values = [sarima_fit_updated.aic, sarima_fit_updated.bic]
criteria_names = ["AIC", "BIC"]
axes[2].bar(criteria_names, aic_bic_values, color=['blue', 'red'])
axes[2].set_title('Model Information Criteria')
axes[2].set_ylabel('Value')
for i, v in enumerate(aic_bic_values):
    axes[2].text(i, v + 5, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
axes[2].grid(axis='y')

# Save the plots to a file
plot_file_path = "validation_plots.png"
plt.tight_layout()
plt.savefig(plot_file_path)
plt.close()

# Plot the residuals and the normal Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Residuals plot
axes[0].plot(updated_df['Date'][:len(residuals_updated)], residuals_updated, color='green')
axes[0].set_title('Residuals Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residuals')
axes[0].grid(True)

# Normal Q-Q Plot of Residuals
sm.qqplot(residuals_updated, line='45', fit=True, ax=axes[1])
axes[1].set_title('Normal Q-Q Plot of Residuals')
axes[1].grid(True)

plt.tight_layout()

qq_plot_file_path = "normalqq.png"
plt.tight_layout()
plt.savefig(qq_plot_file_path)
plt.close()

forecast_file_path_updated, validation_file_path, plot_file_path, qq_plot_file_path
