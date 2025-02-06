# Pfizer Stock Analysis and Global Health Events

## Project Overview
This project explores the relationship between global health events and Pfizer's stock performance. The aim is to define how global health trends, such as vaccination rates and pandemic events, influence Pfizer’s stock returns.

## Objectives
- Analyze stock performance using Yahoo Finance data.
- Incorporate global health data from WHO and other sources.
- Visualize key trends and build predictive signals for stock behavior.

## Getting Started
1. Clone this repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the Streamlit app with:
   ```bash
   streamlit run main.py
   ```
4. Explore the interactive analysis using the provided buttons in the app.

## Directory Structure
```
src/        # Scripts and utility functions
data/       # Raw and processed datasets
tests/      # Test scripts
visualizations/       # plots and graphs
```

## Requirements
- Python 3.x
- Required libraries: pandas, numpy, matplotlib, seaborn, jupyter, streamlit

## Features
- **Interactive Data Analysis**: Run different parts of the analysis by clicking buttons.
- **Event-Based Filtering**: Select window sizes to filter stock data around global health events.
- **Autocorrelation & Feature Importance**: Identify key predictors of stock movement.
- **Regression & Multicollinearity Analysis**: Perform regressions and check for multicollinearity issues.

