 Data Structure
 dataset has 52 columns that include:

Game information (IDs, dates)
Team details (home/away status, scores)
Betting odds (moneylines, spreads, totals)
Line movement metrics (odds changes, volatility)
Outcome indicators (who won, who covered the spread)

The Analysis Pipeline
The script I've created contains four main components:
1. Exploratory Data Analysis (EDA)
The perform_eda() function will:

Show basic statistics about your dataset
Analyze missing values
Examine the distribution of betting outcomes
Create visualizations including:

Missing values heatmap
Home vs. away win distribution
Spread vs. actual score difference
Correlation matrix of key betting features
Line movement vs. outcomes



2. Data Preprocessing
The preprocess_data() function will:

Convert date columns to datetime format and extract useful features
Select relevant numeric features, including line movement metrics
Handle missing values through imputation
Split the data into training and testing sets
Scale the features for better model performance

3. Model Building
The build_models() function will:

Train three types of models:

Logistic Regression
Random Forest
Gradient Boosting


Evaluate each model using accuracy, ROC AUC, and classification reports
Generate confusion matrices and feature importance plots
Compare the models and identify the best performer

4. Betting Strategy Analysis
The analyze_betting_strategies() function will:

Calculate prediction accuracy at different confidence levels
Simulate three betting strategies:

Standard (bet on all predictions)
High confidence (bet only when the model is very confident)
Contrarian (bet against public sentiment using line movement data)


Calculate ROI and total profit for each strategy
Compare strategies visually
