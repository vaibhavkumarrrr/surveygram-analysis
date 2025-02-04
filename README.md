# SurveyGram Report Generator

## Overview

SurveyGram is an automated data analysis and reporting tool designed to
process survey datasets, generate visualizations, and perform predictive
modeling. It generates detailed profiling reports and predictive
analysis insights using linear regression, and exports results as a
comprehensive PDF report.

## Features

-   **Data Profiling**: Generates an extensive profiling report of the
    dataset.
-   **Visualization Generation**: Automatically creates scatter plots,
    line plots, and pair plots.
-   **Data Preprocessing**: Handles categorical encoding, feature
    scaling, and train-test splitting.
-   **Predictive Modeling**: Implements Linear Regression for price
    prediction.
-   **Evaluation Metrics**: Computes Mean Squared Error (MSE) and
    R-squared (R²) scores.
-   **PDF Report Generation**: Combines visualizations and results into
    a structured PDF report.

## Dependencies

Ensure you have the following Python packages installed:

``` sh
pip install pandas seaborn matplotlib ydata-profiling scikit-learn reportlab scipy pillow
```

## Usage

### 1. Load Data

Ensure your dataset (CSV format) is available. The script expects a
dataset similar to `nissan-dataset.csv`.

### 2. Run the Report Generator

``` python
import pandas as pd
from surveygram import SurveyGramReport

df = pd.read_csv('nissan-dataset.csv')
report_generator = SurveyGramReport(df, image_folder='images', pdf_name='my_report.pdf')
report_pdf = report_generator.generate_report()
print(f"Report generated: {report_pdf}")
```

### 3. Output

-   A profiling HTML report (`profiling_report.html`)
-   A set of generated images (`images/` folder)
-   A structured PDF report (`my_report.pdf`)

## Methods Breakdown

-   `profile_data()`: Generates an HTML profiling report.
-   `create_visualizations()`: Produces scatter, line, and pair plots.
-   `preprocess_data(test_size, random_state)`: Encodes and scales data
    for modeling.
-   `linear_regression(X_train, X_test, y_train, y_test)`: Trains and
    evaluates a linear regression model.
-   `evaluate_model(y_true, y_pred)`: Computes MSE and R² scores.
-   `create_report_pdf(mse, r2)`: Consolidates results and images into a
    PDF report.
-   `generate_report()`: Calls all necessary methods to produce the
    final output.

## Output Example

    Report generated: my_report.pdf

## License

This project is licensed under the MIT License.
