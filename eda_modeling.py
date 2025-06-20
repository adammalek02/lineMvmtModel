import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def perform_eda(df):
    """
    Perform exploratory data analysis on the NFL betting dataset with additional insights
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    """
    print("=" * 50)
    print("NFL BETTING DATASET EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # 1. Basic dataset information
    print("\n1. Dataset Overview:")
    print(f"Shape: {df.shape}")
    
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        print(f"Time period: {df['game_date'].min()} to {df['game_date'].max()}")
    
    print(f"Number of unique games: {df['game_id'].nunique() if 'game_id' in df.columns else 'N/A'}")
    print(f"Number of unique teams: {df['api_team_id'].nunique() if 'api_team_id' in df.columns else 'N/A'}")
    
    # 2. Check for missing values
    print("\n2. Missing Values Analysis:")
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    
    print("Columns with missing values:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # 3. Data types analysis
    print("\n3. Data Types:")
    print(df.dtypes.value_counts())
    
    # 4. Betting outcomes distribution
    print("\n4. Betting Outcomes Distribution:")
    betting_outcomes = {}
    
    outcome_cols = ['home_team_won', 'away_team_won', 'over_hit', 'under_hit', 
                   'home_covered_spread', 'away_covered_spread', 'this_team_covered']
    
    for col in outcome_cols:
        if col in df.columns:
            betting_outcomes[col] = df[col].mean()
    
    print(pd.Series(betting_outcomes))
    
    # 5. Line movement analysis
    print("\n5. Line Movement Analysis:")
    movement_cols = ['home_odds_change', 'away_odds_change', 'home_odds_pct_change', 
                     'away_odds_pct_change', 'movement_count']
    
    movement_stats = {}
    for col in movement_cols:
        if col in df.columns:
            movement_stats[f'Avg {col}'] = df[col].mean()
    
    if movement_stats:
        print(pd.Series(movement_stats))
    else:
        print("Line movement columns not found in the dataset.")
    
    # 6. Temporal analysis
    if 'game_date' in df.columns:
        print("\n6. Temporal Analysis:")
        df['year'] = df['game_date'].dt.year
        yearly_stats = df.groupby('year')['this_team_covered'].mean() if 'this_team_covered' in df.columns else None
        
        if yearly_stats is not None:
            print("Cover rate by year:")
            print(yearly_stats)
            
            plt.figure(figsize=(10, 6))
            yearly_stats.plot(kind='bar')
            plt.title('Cover Rate by Year')
            plt.ylabel('Cover Rate')
            plt.tight_layout()
            plt.savefig('cover_rate_by_year.png')
    
    # 7. Check for class imbalance
    if 'this_team_covered' in df.columns:
        print("\n7. Class Balance Analysis:")
        cover_dist = df['this_team_covered'].value_counts(normalize=True)
        print("Class distribution:")
        print(cover_dist)
        
        plt.figure(figsize=(8, 5))
        cover_dist.plot(kind='bar')
        plt.title('Class Distribution')
        plt.ylabel('Percentage')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
    
    # 8. Correlation analysis of key features
    print("\n8. Correlation Analysis:")
    # Exclude post-game result features that would cause data leakage
    exclude_cols = ['api_team_score', 'opponent_score', 'home_team_won', 'away_team_won',
                   'home_score', 'away_score', 'over_hit', 'under_hit', 
                   'home_covered_spread', 'away_covered_spread','spreadWinner','moneylineWinner','home_winner', 'away_winner']
    
    # Select potential predictor columns (numeric only)
    predictor_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                     if col not in exclude_cols and 'score' not in col.lower()]
    
    # Make sure we never treat the target as a predictor
    if 'this_team_covered' in predictor_cols:
        predictor_cols.remove('this_team_covered')

    # Add target variable if it exists
    if 'this_team_covered' in df.columns:
        correlation_cols = predictor_cols + ['this_team_covered']
    else:
        correlation_cols = predictor_cols
    
    if correlation_cols:
        # Calculate correlations
        correlation_matrix = df[correlation_cols].corr()
        
        # Print correlations with target
        if 'this_team_covered' in correlation_cols:
            print("Correlations with target variable:")
            # FIX: For Series, no need to specify 'by' parameter
            target_corr = correlation_matrix['this_team_covered'].sort_values(ascending=False)
            print(target_corr)
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix of Key Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
    
    return df

def preprocess_data(df, target_col, test_size=0.2):
    """
    Preprocess the NFL betting dataset for modeling with improved techniques
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to preprocess
    target_col : str, optional (default='this_team_covered')
        The target column for prediction
    test_size : float, optional (default=0.2)
        The proportion of data to use for testing
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        The preprocessed training and testing data
    """
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)
    
    # Make a copy of the dataframe to avoid modifying the original
    df_prep = df.copy()
    
    # 1. Extract date features if present
    if 'game_date' in df_prep.columns:
        df_prep['game_date'] = pd.to_datetime(df_prep['game_date'])
        df_prep['game_year'] = df_prep['game_date'].dt.year
        df_prep['game_month'] = df_prep['game_date'].dt.month
        df_prep['game_day'] = df_prep['game_date'].dt.day
        df_prep['game_dayofweek'] = df_prep['game_date'].dt.dayofweek
    
    # 2. Remove post-game features that would cause data leakage
    leakage_features = [
        'api_team_score', 'opponent_score', 'home_team_won', 'away_team_won',
        'home_score', 'away_score', 'over_hit', 'under_hit', 
        'home_covered_spread', 'away_covered_spread','spreadwinner','moneylineWinner','home_winner','away_winner' 
    ]
    
    for feature in leakage_features:
        if feature in df_prep.columns and feature != target_col:
            df_prep = df_prep.drop(columns=[feature])
            print(f"Dropped leakage feature: {feature}")
    
    # 3. Select relevant pre-game features
    important_features = [
        'spread', 'overOdds', 'underOdds', 'totalLine', 'moneyLineOdds', 'spreadOdds',
        'is_home', 'is_away','api_team_id'
    ]
    
    movement_features = [
        'start_home_odds', 'end_home_odds', 'start_away_odds', 'end_away_odds',
        'home_odds_change', 'away_odds_change', 'home_odds_pct_change', 'away_odds_pct_change',
        'home_odds_std', 'away_odds_std', 'home_direction_changes', 'away_direction_changes',
        'movement_count'
    ]
    
    # Only include existing columns
    selected_features = [col for col in important_features if col in df_prep.columns]
    selected_features += [col for col in movement_features if col in df_prep.columns]
    
    # Add calendar features if they exist and were created
    calendar_features = ['game_year', 'game_month', 'game_day', 'game_dayofweek']
    selected_features += [col for col in calendar_features if col in df_prep.columns]
    
    # Display selected features
    print(f"\nSelected {len(selected_features)} features for modeling:")
    print(selected_features)
    
    # 4. Handle missing values
    df_clean = df_prep.copy()
    
    # Drop rows with too many missing values in selected features
    missing_count = df_clean[selected_features].isnull().sum(axis=1)
    too_many_missing = missing_count > len(selected_features) * 0.3  # If more than 30% features are missing
    
    print(f"\nDropping {too_many_missing.sum()} rows with too many missing values")
    df_clean = df_clean[~too_many_missing]
    
    # Impute remaining missing values with median
    for feature in selected_features:
        if feature in df_clean.columns and df_clean[feature].isnull().any():
            median_value = df_clean[feature].median()
            df_clean[feature] = df_clean[feature].fillna(median_value)
            print(f"Imputed missing values in {feature} with median ({median_value:.4f})")
    
    # 5. Split data chronologically if date column exists
    if 'game_date' in df_clean.columns:
        print("\nSplitting data chronologically...")
        df_clean = df_clean.sort_values('game_date')
        train_idx = int(len(df_clean) * (1 - test_size))
        
        train_data = df_clean.iloc[:train_idx]
        test_data = df_clean.iloc[train_idx:]
        
        X_train = train_data[selected_features]
        y_train = train_data[target_col]
        X_test = test_data[selected_features]
        y_test = test_data[target_col]
        
        print(f"Training data: {len(train_data)} samples from {train_data['game_date'].min()} to {train_data['game_date'].max()}")
        print(f"Testing data: {len(test_data)} samples from {test_data['game_date'].min()} to {test_data['game_date'].max()}")
    else:
        print("\nSplitting data randomly...")
        # If no date column, fall back to random split
        X = df_clean[selected_features]
        y = df_clean[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Testing data: {len(X_test)} samples")
    
    # 6. Check class balance
    print("\nClass distribution in training data:")
    print(y_train.value_counts(normalize=True))
    
    print("\nClass distribution in testing data:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test, selected_features

# --- Individual Model Building Functions ---

def build_logistic_regression(X_train, X_test, y_train, y_test, features, cv=5, model_name="Logistic Regression", custom_params=None):
    """
    Build and evaluate a Logistic Regression model for NFL betting prediction
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        The preprocessed training and testing features
    y_train, y_test : pandas.Series
        The target variables for training and testing
    features : list
        List of feature names
    cv : int, optional (default=5)
        Number of cross-validation folds
    model_name : str, optional (default="Logistic Regression")
        Name of the model for display purposes
    custom_params : dict, optional (default=None)
        Custom parameters for the model
        
    Returns:
    --------
    model : sklearn Pipeline
        The trained model
    results : dict
        Dictionary containing model performance metrics
    """
    print(f"\nBuilding {model_name} model...")
    
    # Default parameters
    lr_params = {'max_iter': 1000, 'random_state': 42, 'C': 0.1}
    
    # Update with custom parameters if provided
    if custom_params:
        lr_params.update(custom_params)
    
    # Create pipeline with standardization
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**lr_params))
    ])
    
    # Cross-validation
    cv_split = TimeSeriesSplit(n_splits=cv)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_split, scoring='roc_auc')
    print(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    # Create precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'precision_recall_{model_name.replace(" ", "_").lower()}.png')
    
    # Feature importance (using coefficients for logistic regression)
    coefficients = model.named_steps['classifier'].coef_[0]
    abs_coeffs = np.abs(coefficients)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': abs_coeffs,
        'Direction': np.sign(coefficients)
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    colors = ['red' if direction < 0 else 'blue' for direction in top_features['Direction']]
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient Magnitude')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
    
    print(f"Top 10 Important Features ({model_name}):")
    print(importance_df.head(10))
    
    # Store results
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'feature_importance': importance_df
    }
    
    return model, results

def build_random_forest(X_train, X_test, y_train, y_test, features, cv=5, model_name="Random Forest", custom_params=None):
    """
    Build and evaluate a Random Forest model for NFL betting prediction
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        The preprocessed training and testing features
    y_train, y_test : pandas.Series
        The target variables for training and testing
    features : list
        List of feature names
    cv : int, optional (default=5)
        Number of cross-validation folds
    model_name : str, optional (default="Random Forest")
        Name of the model for display purposes
    custom_params : dict, optional (default=None)
        Custom parameters for the model
        
    Returns:
    --------
    model : sklearn Pipeline
        The trained model
    results : dict
        Dictionary containing model performance metrics
    """
    print(f"\nBuilding {model_name} model...")
    
    # Default parameters
    rf_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    
    # Update with custom parameters if provided
    if custom_params:
        rf_params.update(custom_params)
    
    # Create pipeline with standardization
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**rf_params))
    ])
    
    # Cross-validation
    cv_split = TimeSeriesSplit(n_splits=cv)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_split, scoring='roc_auc')
    print(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    # Create precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'precision_recall_{model_name.replace(" ", "_").lower()}.png')
    
    # Feature importance
    feature_importances = model.named_steps['classifier'].feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
    
    print(f"Top 10 Important Features ({model_name}):")
    print(importance_df.head(10))
    
    # Store results
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'feature_importance': importance_df
    }
    
    return model, results

def build_gradient_boosting(X_train, X_test, y_train, y_test, features, cv=5, model_name="Gradient Boosting", custom_params=None):
    """
    Build and evaluate a Gradient Boosting model for NFL betting prediction
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        The preprocessed training and testing features
    y_train, y_test : pandas.Series
        The target variables for training and testing
    features : list
        List of feature names
    cv : int, optional (default=5)
        Number of cross-validation folds
    model_name : str, optional (default="Gradient Boosting")
        Name of the model for display purposes
    custom_params : dict, optional (default=None)
        Custom parameters for the model
        
    Returns:
    --------
    model : sklearn Pipeline
        The trained model
    results : dict
        Dictionary containing model performance metrics
    """
    print(f"\nBuilding {model_name} model...")
    
    # Default parameters
    gb_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}
    
    # Update with custom parameters if provided
    if custom_params:
        gb_params.update(custom_params)
    
    # Create pipeline with standardization
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(**gb_params))
    ])
    
    # Cross-validation
    cv_split = TimeSeriesSplit(n_splits=cv)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_split, scoring='roc_auc')
    print(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    # Create precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'precision_recall_{model_name.replace(" ", "_").lower()}.png')
    
    # Feature importance
    feature_importances = model.named_steps['classifier'].feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
    
    print(f"Top 10 Important Features ({model_name}):")
    print(importance_df.head(10))
    
    # Store results
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'feature_importance': importance_df
    }
    
    return model, results

# --- Hyperparameter Tuning Functions ---

def tune_logistic_regression(X_train, y_train, cv=5, random_search=True, n_iter=20):
    """
    Perform hyperparameter tuning for Logistic Regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature data
    y_train : pandas.Series
        Training target data
    cv : int, optional (default=5)
        Number of cross-validation folds
    random_search : bool, optional (default=True)
        Whether to use RandomizedSearchCV instead of GridSearchCV
    n_iter : int, optional (default=20)
        Number of parameter settings sampled (for RandomizedSearchCV)
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    """
    print("\nPerforming hyperparameter tuning for Logistic Regression...")
    
    # Create pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__penalty': ['l1', 'l2', 'elasticnet'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__class_weight': [None, 'balanced'],
        'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only for elasticnet
    }
    
    # Create CV splitter
    cv_split = TimeSeriesSplit(n_splits=cv)
    
    # Choose search method
    if random_search:
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1,
            n_iter=n_iter, random_state=42
        )
    else:
        search = GridSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {search.best_params_}")
    print(f"Best ROC AUC: {search.best_score_:.4f}")
    
    # Print top parameter combinations
    results_df = pd.DataFrame(search.cv_results_)
    top_results = results_df.sort_values('mean_test_score', ascending=False).head(5)
    print("\nTop 5 parameter combinations:")
    for i, row in top_results.iterrows():
        params = {k.split('__')[1]: v for k, v in row['params'].items() if k.startswith('classifier__')}
        print(f"Parameters: {params}, ROC AUC: {row['mean_test_score']:.4f}")
    
    # Extract best parameters
    best_params = {k.split('__')[1]: v for k, v in search.best_params_.items() if k.startswith('classifier__')}
    
    return best_params

def tune_random_forest(X_train, y_train, cv=5, random_search=True, n_iter=20):
    """
    Perform hyperparameter tuning for Random Forest model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature data
    y_train : pandas.Series
        Training target data
    cv : int, optional (default=5)
        Number of cross-validation folds
    random_search : bool, optional (default=True)
        Whether to use RandomizedSearchCV instead of GridSearchCV
    n_iter : int, optional (default=20)
        Number of parameter settings sampled (for RandomizedSearchCV)
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    """
    print("\nPerforming hyperparameter tuning for Random Forest...")
    
    # Create pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Create CV splitter
    cv_split = TimeSeriesSplit(n_splits=cv)
    
    # Choose search method
    if random_search:
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1,
            n_iter=n_iter, random_state=42
        )
    else:
        search = GridSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {search.best_params_}")
    print(f"Best ROC AUC: {search.best_score_:.4f}")
    
    # Print top parameter combinations
    results_df = pd.DataFrame(search.cv_results_)
    top_results = results_df.sort_values('mean_test_score', ascending=False).head(5)
    print("\nTop 5 parameter combinations:")
    for i, row in top_results.iterrows():
        params = {k.split('__')[1]: v for k, v in row['params'].items() if k.startswith('classifier__')}
        print(f"Parameters: {params}, ROC AUC: {row['mean_test_score']:.4f}")
    
    # Extract best parameters
    best_params = {k.split('__')[1]: v for k, v in search.best_params_.items() if k.startswith('classifier__')}
    
    return best_params

def tune_gradient_boosting(X_train, y_train, cv=5, random_search=True, n_iter=20):
    """
    Perform hyperparameter tuning for Gradient Boosting model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature data
    y_train : pandas.Series
        Training target data
    cv : int, optional (default=5)
        Number of cross-validation folds
    random_search : bool, optional (default=True)
        Whether to use RandomizedSearchCV instead of GridSearchCV
    n_iter : int, optional (default=20)
        Number of parameter settings sampled (for RandomizedSearchCV)
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    """
    print("\nPerforming hyperparameter tuning for Gradient Boosting...")
    
    # Create pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [2, 3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    # Create CV splitter
    cv_split = TimeSeriesSplit(n_splits=cv)
    
    # Choose search method
    if random_search:
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1,
            n_iter=n_iter, random_state=42
        )
    else:
        search = GridSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {search.best_params_}")
    print(f"Best ROC AUC: {search.best_score_:.4f}")
    
    # Print top parameter combinations
    results_df = pd.DataFrame(search.cv_results_)
    top_results = results_df.sort_values('mean_test_score', ascending=False).head(5)
    print("\nTop 5 parameter combinations:")
    for i, row in top_results.iterrows():
        params = {k.split('__')[1]: v for k, v in row['params'].items() if k.startswith('classifier__')}
        print(f"Parameters: {params}, ROC AUC: {row['mean_test_score']:.4f}")
    
    # Extract best parameters
    best_params = {k.split('__')[1]: v for k, v in search.best_params_.items() if k.startswith('classifier__')}
    
    return best_params

# --- Main Model Building Function ---

def build_models(X_train, X_test, y_train, y_test, features, cv=5, tune_hyperparams=False, random_search=True, n_iter=20):
    """
    Build and evaluate machine learning models for NFL betting prediction
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        The preprocessed training and testing features
    y_train, y_test : pandas.Series
        The target variables for training and testing
    features : list
        List of feature names
    cv : int, optional (default=5)
        Number of cross-validation folds
    tune_hyperparams : bool, optional (default=False)
        Whether to perform hyperparameter tuning
    random_search : bool, optional (default=True)
        Whether to use RandomizedSearchCV instead of GridSearchCV
    n_iter : int, optional (default=20)
        Number of parameter settings sampled (for RandomizedSearchCV)
    
    Returns:
    --------
    best_model : sklearn Pipeline
        The best performing model
    models_dict : dict
        Dictionary containing all trained models
    results_dict : dict
        Dictionary containing all model results
    """
    print("\n" + "=" * 50)
    print("NFL BETTING PREDICTION MODELS")
    print("=" * 50)
    
    # Initialize parameters
    lr_params = None
    rf_params = None
    gb_params = None
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparams:
        print("\nPerforming hyperparameter tuning...")
        lr_params = tune_logistic_regression(X_train, y_train, cv, random_search, n_iter)
        rf_params = tune_random_forest(X_train, y_train, cv, random_search, n_iter)
        gb_params = tune_gradient_boosting(X_train, y_train, cv, random_search, n_iter)
    
    # Build individual models
    lr_model, lr_results = build_logistic_regression(
        X_train, X_test, y_train, y_test, features, cv, 
        model_name="Logistic Regression", custom_params=lr_params
    )
    
    rf_model, rf_results = build_random_forest(
        X_train, X_test, y_train, y_test, features, cv,
        model_name="Random Forest", custom_params=rf_params
    )
    
    gb_model, gb_results = build_gradient_boosting(
        X_train, X_test, y_train, y_test, features, cv,
        model_name="Gradient Boosting", custom_params=gb_params
    )
    
    # Combine results
    models_dict = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
    }
    
    results_dict = {
        'Logistic Regression': lr_results,
        'Random Forest': rf_results,
        'Gradient Boosting': gb_results
    }
    
        # --- compare and pick best ---
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results_dict[m]['accuracy'] for m in results_dict],
        'ROC AUC': [results_dict[m]['roc_auc'] for m in results_dict]
    }).sort_values('ROC AUC', ascending=False)
    print(comparison_df)

    # barplot of performance
    plt.figure(figsize=(10, 6))
    comp_melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=comp_melted)
    plt.ylim(0.5, 1.0)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison.png')

    # pick best
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    print(f"\nBest Model: {best_model_name} (ROC AUC = {comparison_df.iloc[0]['ROC AUC']:.4f})")

    return best_model, models_dict, results_dict



'''def build_models(X_train, X_test, y_train, y_test, features):
    """
    Build and evaluate machine learning models for NFL betting prediction with cross-validation
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        The preprocessed training and testing features
    y_train, y_test : pandas.Series
        The target variables for training and testing
    features : list
        List of feature names
    """
    print("\n" + "=" * 50)
    print("NFL BETTING PREDICTION MODELS")
    print("=" * 50)
    
    # 1. Define models to test
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=0.1))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))
        ])
    }
    
    # 2. Cross-validation for each model
    print("\nPerforming cross-validation...")
    cv = TimeSeriesSplit(n_splits=5)  # Time series cross-validation
    
    for name, model in models.items():
        print(f"\n{name} Cross-Validation:")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 3. Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test ROC AUC: {roc_auc:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name}')
        plt.savefig(f'precision_recall_{name.replace(" ", "_").lower()}.png')
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'model': model
        }
        
        # Feature importance using permutation importance
        if hasattr(model[-1], 'feature_importances_'):
            feature_importances = model[-1].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
        else:
            # For models without built-in feature importance, use permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
            
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Feature Importance - {name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
        
        print(f"Top 10 Important Features ({name}):")
        print(importance_df.head(10))
    
    # 4. Compare models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'ROC AUC': [results[model]['roc_auc'] for model in results]
    })
    print(comparison_df.sort_values('ROC AUC', ascending=False))
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    comparison_df_melted = pd.melt(comparison_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_df_melted)
    plt.title('Model Performance Comparison')
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # 5. Return the best model
    best_model_name = comparison_df.iloc[comparison_df['ROC AUC'].argmax()]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    return best_model, results'''

'''def analyze_betting_strategies(X_test, y_test, best_model, threshold=0.55):
    """
    Analyze profitable betting strategies based on model predictions
    
    Parameters:
    -----------
    X_test : pandas.DataFrame
        Test feature data
    y_test : pandas.Series
        Actual outcomes
    best_model : sklearn Pipeline
        The best trained model
    threshold : float, optional (default=0.55)
        Probability threshold for making bets
    """
    print("\n" + "=" * 50)
    print("BETTING STRATEGY ANALYSIS")
    print("=" * 50)
    
    # Get model predictions
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Create a DataFrame for strategy analysis
    strategy_df = pd.DataFrame({
        'actual_outcome': y_test,
        'predicted_prob': y_pred_proba
    })
    
    # Calculate correct predictions at different probability thresholds
    print("\n1. Performance at Different Probability Thresholds:")
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    threshold_results = []
    for thresh in thresholds:
        # Make binary predictions based on threshold
        strategy_df[f'pred_{thresh}'] = (strategy_df['predicted_prob'] >= thresh).astype(int)
        
        # Calculate performance metrics
        coverage = (strategy_df['predicted_prob'] >= thresh).mean()
        bets_made = (strategy_df['predicted_prob'] >= thresh).sum()
        
        if bets_made > 0:
            accuracy = (strategy_df[f'pred_{thresh}'] == strategy_df['actual_outcome']).mean()
            win_rate = strategy_df.loc[strategy_df['predicted_prob'] >= thresh, 'actual_outcome'].mean()
            
            # Calculate profit with -110 odds (bet 110 to win 100)
            wins = strategy_df.loc[strategy_df['predicted_prob'] >= thresh, 'actual_outcome'].sum()
            losses = bets_made - wins
            profit = wins * 100 - losses * 110
            roi = profit / (bets_made * 110) * 100
            
            threshold_results.append({
                'Threshold': thresh,
                'Bets Made': bets_made,
                'Coverage (%)': coverage * 100,
                'Win Rate (%)': win_rate * 100,
                'Accuracy (%)': accuracy * 100,
                'Profit (units)': profit / 110,
                'ROI (%)': roi
            })
    
    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df)
    
    # Plot threshold performance
    plt.figure(figsize=(12, 6))
    plt.plot(threshold_df['Threshold'], threshold_df['ROI (%)'], marker='o', label='ROI (%)')
    plt.plot(threshold_df['Threshold'], threshold_df['Win Rate (%)'], marker='s', label='Win Rate (%)')
    plt.plot(threshold_df['Threshold'], [52.38] * len(threshold_df), 'r--', label='Break-even (52.38%)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Percentage')
    plt.title('Win Rate and ROI by Probability Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('threshold_performance.png')
    
    # 2. Bankroll simulation
    print("\n2. Bankroll Simulation with Optimal Threshold:")
    # Find optimal threshold from our analysis
    optimal_threshold_row = threshold_df.iloc[threshold_df['ROI (%)'].argmax()]
    optimal_threshold = optimal_threshold_row['Threshold']
    
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Expected win rate: {optimal_threshold_row['Win Rate (%)']}%")
    print(f"Expected ROI: {optimal_threshold_row['ROI (%)']}%")
    
    # 3. Kelly Criterion for optimal bet sizing
    win_prob = optimal_threshold_row['Win Rate (%)'] / 100
    odds = 10/11  # -110 odds payout ratio
    
    # Kelly formula: f* = (bp - q) / b where b = odds, p = win probability, q = loss probability
    kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
    
    print(f"\n3. Kelly Criterion Bet Sizing:")
    print(f"Full Kelly: {kelly_fraction:.4f} (bet {kelly_fraction*100:.2f}% of bankroll)")
    print(f"Quarter Kelly (recommended): {kelly_fraction/4:.4f} (bet {kelly_fraction*25:.2f}% of bankroll)")
    
    # 4. Recommended betting strategy
    print("\n4. Recommended Betting Strategy:")
    print(f"- Only bet when model probability >= {optimal_threshold}")
    print(f"- Expected to bet on approximately {optimal_threshold_row['Coverage (%)']}% of games")
    print(f"- Use fractional Kelly sizing: {kelly_fraction/4:.4f} of bankroll per bet")
    print(f"- Expected ROI: {optimal_threshold_row['ROI (%)']}%")
    
    return strategy_df, optimal_threshold
    '''
'''def analyze_betting_strategiesMl(X_test, y_test, best_model, thresholds=[0.5,0.55,0.6,0.65,0.7,0.75]):
    """
    Analyze profitable betting strategies based on model predictions
    using the true money-line odds per game.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # helper to turn American ML odds into units won per 1 unit risk
    def payout_multiplier(ml_odds):
        if ml_odds > 0:
            return ml_odds / 100.0
        else:
            return 100.0 / abs(ml_odds)

    # 1) build a unified DataFrame
    strategy_df = X_test.copy()
    strategy_df['actual_outcome'] = y_test
    strategy_df['predicted_prob'] = best_model.predict_proba(X_test)[:,1]
    strategy_df['payout_mult'] = strategy_df['moneyLineOdds'].apply(payout_multiplier)

    results = []
    for thresh in thresholds:
        bets = strategy_df.loc[strategy_df['predicted_prob'] >= thresh].copy()

## maybe thinking abiyt adding something to visualoize what bets the model is actually making, 

    #considering a spread model fore lines < -125

        n_bets = len(bets)
        if n_bets == 0:
            continue

        # compute win/loss per bet
        # win → +payout, loss → –1 unit staked
        #bets['payout_mult'] = bets['moneyLineOdds'].apply(payout_multiplier)
        bets['profit'] = bets.apply(
            lambda r: r['payout_mult'] if r['actual_outcome']==1 else -1.0,
            axis=1
        )
        total_profit = bets['profit'].sum()
        total_staked = n_bets * 1.0  # 1 unit per bet
        roi = total_profit / total_staked * 100

        win_rate   = bets['actual_outcome'].mean() * 100
        accuracy   = (bets['actual_outcome'] == (bets['predicted_prob']>=thresh)).mean() * 100
        coverage   = n_bets / len(strategy_df) * 100

        results.append({
            'Threshold':      thresh,
            'Bets Made':      n_bets,
            'Coverage (%)':   coverage,
            'Win Rate (%)':   win_rate,
            'Accuracy (%)':   accuracy,
            'Profit (units)': total_profit,
            'ROI (%)':        roi
        })

    threshold_df = pd.DataFrame(results)
    print("\nPerformance at Different Probability Thresholds:")
    print(threshold_df)

    # visualize
    plt.figure(figsize=(10,5))
    plt.plot(threshold_df['Threshold'], threshold_df['ROI (%)'], marker='o', label='ROI (%)')
    plt.plot(threshold_df['Threshold'], threshold_df['Win Rate (%)'], marker='s', label='Win Rate (%)')
    plt.xlabel('Probability Threshold')
    plt.title('Moneyline Bet: Win Rate vs ROI by Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ml_threshold_performance.png')

    # find the best ROI threshold
    best_row = threshold_df.loc[threshold_df['ROI (%)'].idxmax()]
    best_thresh = best_row['Threshold']
    print(f"\n→ Optimal Threshold: {best_thresh} (ROI: {best_row['ROI (%)']:.1f}%, Win Rate: {best_row['Win Rate (%)']:.1f}%)")

    #2) Kelly Criterion using *actual* payout multipliers
    best_bets = strategy_df.loc[strategy_df['predicted_prob'] >= best_thresh].copy()
    p = best_bets['actual_outcome'].mean()             # empirical win prob
    average_b = best_bets['payout_mult'].mean()        # average $ won per $1 risked

    # Kelly formula: f* = (b·p – q) / b
    kelly_fraction = (average_b * p - (1 - p)) / average_b

    print(f"\nKelly Criterion Bet Sizing (using true ML odds):")
    print(f"- Empirical p: {p:.3f}, avg payout b: {average_b:.3f}")
    print(f"- Full Kelly fraction: {kelly_fraction:.3f} of bankroll")
    print(f"- 1/4 Kelly fraction: {kelly_fraction/4:.3f} of bankroll")


    # 2. Bankroll simulation
    print("\n2. Bankroll Simulation with Optimal Threshold:")
    # Find optimal threshold from our analysis
    optimal_threshold_row = threshold_df.iloc[threshold_df['ROI (%)'].argmax()]
    optimal_threshold = optimal_threshold_row['Threshold']
    
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Expected win rate: {optimal_threshold_row['Win Rate (%)']}%")
    print(f"Expected ROI: {optimal_threshold_row['ROI (%)']}%")
    

    # 3. Kelly Criterion for optimal bet sizing
    best_bets = strategy_df.loc[strategy_df['predicted_prob'] >= optimal_threshold].copy()

    # empirical win probability
    p = best_bets['actual_outcome'].mean()

    # true average payout per $1 risked
    # (payout_mult already = ml_odds/100  if positive, or 100/|ml_odds| if negative)
    b = best_bets['payout_mult'].mean()

    # Kelly formula: f* = (b*p - q) / b, where q = 1 - p
    kelly_fraction = (b * p - (1 - p)) / b

    # 4. Recommended betting strategy
    print(f"\n3. Kelly Criterion Bet Sizing (using true ML odds):")
    print(f"- Empirical win prob p: {p:.3f}")
    print(f"- Avg payout multiplier b: {b:.3f}")
    print(f"- Full Kelly fraction: {kelly_fraction:.3f} of bankroll")
    print(f"- 1/4 Kelly fraction: {kelly_fraction/4:.3f} of bankroll")
    
    print("\n4. Recommended Betting Strategy:")
    print(f"- Only bet when model probability >= {optimal_threshold}")
    print(f"- Expected to bet on approximately {optimal_threshold_row['Coverage (%)']}% of games")
    print(f"- Use fractional Kelly sizing: {kelly_fraction/4:.4f} of bankroll per bet")
    print(f"- Expected ROI: {optimal_threshold_row['ROI (%)']}%")

    return strategy_df, threshold_df, best_thresh'''
def analyze_betting_strategiesMl(X_test, y_test, best_model, thresholds=[0.5,0.55,0.6,0.65,0.7,0.75], show_detailed_bets=True):
    """
    Analyze profitable betting strategies based on model predictions
    using the true money-line odds per game.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # helper to turn American ML odds into units won per 1 unit risk
    def payout_multiplier(ml_odds):
        if ml_odds > 0:
            return ml_odds / 100.0
        else:
            return 100.0 / abs(ml_odds)

    # 1) build a unified DataFrame
    strategy_df = X_test.copy()
    strategy_df['actual_outcome'] = y_test
    strategy_df['predicted_prob'] = best_model.predict_proba(X_test)[:,1]
    strategy_df['payout_mult'] = strategy_df['moneyLineOdds'].apply(payout_multiplier)

    # Add a column to classify if it's a favorite or underdog
    strategy_df['is_favorite'] = strategy_df['moneyLineOdds'] < 0
    
    # Add detailed print of moneyline odds range to help with debugging
    print(f"MoneyLine Odds range: Min={strategy_df['moneyLineOdds'].min()}, Max={strategy_df['moneyLineOdds'].max()}")
    
    try:
        strategy_df['odds_category'] = pd.cut(
            strategy_df['moneyLineOdds'], 
            bins=[-10000, -300, -200, -150, -110, 0, 110, 150, 200, 300, 10000],
            labels=['Heavy Fav (-300+)', 'Strong Fav (-300 to -200)', 'Moderate Fav (-200 to -150)', 
                    'Slight Fav (-150 to -110)', 'Pick\'em', 'Slight Dog (+100 to +110)', 
                    'Moderate Dog (+110 to +150)', 'Strong Dog (+150 to +200)', 
                    'Heavy Dog (+200 to +300)', 'Extreme Dog (+300+)']
        )
    except Exception as e:
        print(f"Warning: Could not categorize odds: {str(e)}")
        # Create a simpler version as fallback
        try:
            strategy_df['odds_category'] = np.where(strategy_df['moneyLineOdds'] < -200, 'Heavy Favorite',
                                         np.where(strategy_df['moneyLineOdds'] < 0, 'Favorite',
                                         np.where(strategy_df['moneyLineOdds'] < 200, 'Underdog', 'Heavy Underdog')))
        except Exception as e2:
            print(f"Warning: Could not create simple odds categories either: {str(e2)}")
            # We'll proceed without the odds_category column

    results = []
    odds_distributions = {}
    
    for thresh in thresholds:
        bets = strategy_df.loc[strategy_df['predicted_prob'] >= thresh].copy()
        
        n_bets = len(bets)
        if n_bets == 0:
            continue

        # Store odds distribution data for this threshold
        odds_dist = {
            'moneyline_odds': bets['moneyLineOdds'].values,
            'fav_pct': (bets['is_favorite'].mean() * 100),
            'dog_pct': ((~bets['is_favorite']).mean() * 100),
            'avg_odds': bets['moneyLineOdds'].mean(),
            'median_odds': bets['moneyLineOdds'].median()
        }
        
        # Only add odds_category if it exists in the DataFrame
        if 'odds_category' in bets.columns:
            try:
                odds_dist['odds_category_counts'] = bets['odds_category'].value_counts().to_dict()
            except Exception as e:
                print(f"Warning: Could not count odds categories for threshold {thresh}: {str(e)}")
                odds_dist['odds_category_counts'] = {}
        
        odds_distributions[thresh] = odds_dist

        # compute win/loss per bet
        bets['profit'] = bets.apply(
            lambda r: r['payout_mult'] if r['actual_outcome']==1 else -1.0,
            axis=1
        )
        total_profit = bets['profit'].sum()
        total_staked = n_bets * 1.0  # 1 unit per bet
        roi = total_profit / total_staked * 100

        win_rate   = bets['actual_outcome'].mean() * 100
        accuracy   = (bets['actual_outcome'] == (bets['predicted_prob']>=thresh)).mean() * 100
        coverage   = n_bets / len(strategy_df) * 100

        results.append({
            'Threshold':      thresh,
            'Bets Made':      n_bets,
            'Coverage (%)':   coverage,
            'Win Rate (%)':   win_rate,
            'Accuracy (%)':   accuracy,
            'Profit (units)': total_profit,
            'ROI (%)':        roi,
            'Avg ML Odds':    odds_distributions[thresh]['avg_odds'],
            'Median ML Odds': odds_distributions[thresh]['median_odds'],
            'Favs (%)':       odds_distributions[thresh]['fav_pct'],
            'Dogs (%)':       odds_distributions[thresh]['dog_pct']
        })

    threshold_df = pd.DataFrame(results)
    print("\nPerformance at Different Probability Thresholds:")
    print(threshold_df)

    # Print detailed odds distribution for each threshold
    print("\nDetailed Moneyline Odds Distribution by Threshold:")
    for thresh, dist_data in odds_distributions.items():
        print(f"\nThreshold: {thresh}")
        print(f"Average ML Odds: {dist_data['avg_odds']:.1f}")
        print(f"Median ML Odds: {dist_data['median_odds']:.1f}")
        print(f"Favorites: {dist_data['fav_pct']:.1f}%, Underdogs: {dist_data['dog_pct']:.1f}%")
        
        # Print individual moneyline odds for each bet at this threshold
        bets = strategy_df.loc[strategy_df['predicted_prob'] >= thresh].copy()
        print(f"All moneyline odds for {len(bets)} bets at threshold {thresh}:")
        odds_list = bets['moneyLineOdds'].sort_values().tolist()
        # Print in groups of 10 for readability
        for i in range(0, len(odds_list), 10):
            group = odds_list[i:i+10]
            print(f"  {group}")
        
        # Print odds category breakdown if available
        if 'odds_category_counts' in dist_data and dist_data['odds_category_counts']:
            try:
                print("Odds Category Breakdown:")
                total_bets = sum(dist_data['odds_category_counts'].values())
                if total_bets > 0:
                    for category, count in sorted(dist_data['odds_category_counts'].items(), 
                                              key=lambda x: str(x[0]) if isinstance(x[0], str) else x[0]):
                        pct = (count / total_bets) * 100
                        print(f"  {category}: {count} bets ({pct:.1f}%)")
            except Exception as e:
                print(f"  Warning: Could not print odds category breakdown: {str(e)}")

    # visualize odds distribution for best threshold
    best_row = threshold_df.loc[threshold_df['ROI (%)'].idxmax()]
    best_thresh = best_row['Threshold']
    
    # Visualize ROI vs Threshold
    plt.figure(figsize=(10,5))
    plt.plot(threshold_df['Threshold'], threshold_df['ROI (%)'], marker='o', label='ROI (%)')
    plt.plot(threshold_df['Threshold'], threshold_df['Win Rate (%)'], marker='s', label='Win Rate (%)')
    plt.xlabel('Probability Threshold')
    plt.title('Moneyline Bet: Win Rate vs ROI by Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ml_threshold_performance.png')
    
    # Create histogram of moneyline odds for best threshold
    best_bets = strategy_df.loc[strategy_df['predicted_prob'] >= best_thresh].copy()
    try:
        if len(best_bets) > 0:
            plt.figure(figsize=(12, 6))
            plt.hist(best_bets['moneyLineOdds'], bins=20, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.xlabel('Moneyline Odds')
            plt.ylabel('Number of Bets')
            plt.title(f'Distribution of Moneyline Odds (Threshold = {best_thresh})')
            plt.grid(True, alpha=0.3)
            
            # Add text showing summary statistics
            stats_text = (f"Total Bets: {len(best_bets)}\n"
                          f"Avg Odds: {best_bets['moneyLineOdds'].mean():.1f}\n"
                          f"Median Odds: {best_bets['moneyLineOdds'].median():.1f}\n"
                          f"Favorites: {best_bets['is_favorite'].mean()*100:.1f}%\n"
                          f"Underdogs: {(~best_bets['is_favorite']).mean()*100:.1f}%")
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            plt.savefig('best_threshold_odds_distribution.png')
        else:
            print("Warning: No bets at best threshold, cannot create histogram")
    except Exception as e:
        print(f"Warning: Could not create histogram: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create a boxplot of moneyline odds for each threshold
    try:
        plt.figure(figsize=(14, 7))
        boxplot_data = []
        boxplot_labels = []
        
        for thresh in thresholds:
            if thresh in odds_distributions and len(odds_distributions[thresh]['moneyline_odds']) > 0:
                boxplot_data.append(odds_distributions[thresh]['moneyline_odds'])
                boxplot_labels.append(f"{thresh}")
        
        if boxplot_data:  # Only plot if we have data
            plt.boxplot(boxplot_data, labels=boxplot_labels)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Threshold')
            plt.ylabel('Moneyline Odds')
            plt.title('Distribution of Moneyline Odds by Threshold')
            plt.grid(True, alpha=0.3)
            plt.savefig('threshold_odds_boxplot.png')
        else:
            print("Warning: Not enough data to create boxplot of odds by threshold")
    except Exception as e:
        print(f"Warning: Could not create boxplot: {str(e)}")
        import traceback
        traceback.print_exc()
    
            # Visualize Win Rate by Odds Category for best threshold
    try:
        if len(best_bets) > 0 and 'odds_category' in best_bets.columns:
            plt.figure(figsize=(14, 7))
            
            # Make sure we have valid data to group by
            if not best_bets['odds_category'].isna().all():
                win_rates = best_bets.groupby('odds_category')['actual_outcome'].mean() * 100
                counts = best_bets['odds_category'].value_counts()
                
                # Check if we have valid results
                if not win_rates.empty and not counts.empty:
                    # Create a DataFrame with the data
                    win_rate_df = pd.DataFrame({
                        'Win Rate (%)': win_rates,
                        'Count': counts
                    })
                    # Ensure index name is preserved during reset
                    win_rate_df = win_rate_df.reset_index()
                    
                    # Sort by index name which should now be a column
                    if 'index' in win_rate_df.columns:
                        win_rate_df = win_rate_df.sort_values('index')
                        x_col = 'index'
                    elif 'odds_category' in win_rate_df.columns:
                        win_rate_df = win_rate_df.sort_values('odds_category')
                        x_col = 'odds_category'
                    else:
                        # If we can't find a proper column to sort by, just use the first column
                        x_col = win_rate_df.columns[0]
                    
                    ax = sns.barplot(x=x_col, y='Win Rate (%)', data=win_rate_df, palette='viridis')
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f'Win Rate by Odds Category (Threshold = {best_thresh})')
                    plt.grid(True, axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    # Add count labels on top of bars
                    for i, p in enumerate(ax.patches):
                        if i < len(win_rate_df):
                            ax.annotate(f"n={win_rate_df.iloc[i]['Count']}", 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha = 'center', va = 'bottom',
                                        xytext = (0, 5), textcoords = 'offset points')
                    
                    plt.savefig('win_rate_by_odds_category.png')
                else:
                    print("Warning: Not enough data to visualize win rates by odds category")
            else:
                print("Warning: All odds_category values are NA")
        else:
            print("Warning: No bets at best threshold or odds_category column missing")
    except Exception as e:
        print(f"Warning: Could not create win rate by odds category chart: {str(e)}")
        import traceback
        traceback.print_exc()

    # find the best ROI threshold
    print(f"\n→ Optimal Threshold: {best_thresh} (ROI: {best_row['ROI (%)']:.1f}%, Win Rate: {best_row['Win Rate (%)']:.1f}%)")
    print(f"  Avg ML Odds: {best_row['Avg ML Odds']:.1f}, Favorites: {best_row['Favs (%)']:.1f}%, Underdogs: {best_row['Dogs (%)']:.1f}%")

    #2) Kelly Criterion using *actual* payout multipliers
    p = best_bets['actual_outcome'].mean()             # empirical win prob
    average_b = best_bets['payout_mult'].mean()        # average $ won per $1 risked

    # Kelly formula: f* = (b·p – q) / b
    kelly_fraction = (average_b * p - (1 - p)) / average_b

    print(f"\nKelly Criterion Bet Sizing (using true ML odds):")
    print(f"- Empirical p: {p:.3f}, avg payout b: {average_b:.3f}")
    print(f"- Full Kelly fraction: {kelly_fraction:.3f} of bankroll")
    print(f"- 1/4 Kelly fraction: {kelly_fraction/4:.3f} of bankroll")

    # 2. Bankroll simulation
    print("\n2. Bankroll Simulation with Optimal Threshold:")
    # Find optimal threshold from our analysis
    optimal_threshold_row = threshold_df.iloc[threshold_df['ROI (%)'].argmax()]
    optimal_threshold = optimal_threshold_row['Threshold']
    
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Expected win rate: {optimal_threshold_row['Win Rate (%)']}%")
    print(f"Expected ROI: {optimal_threshold_row['ROI (%)']}%")
    
    # 3. Kelly Criterion for optimal bet sizing
    best_bets = strategy_df.loc[strategy_df['predicted_prob'] >= optimal_threshold].copy()

    # empirical win probability
    p = best_bets['actual_outcome'].mean()

    # true average payout per $1 risked
    b = best_bets['payout_mult'].mean()

    # Kelly formula: f* = (b*p - q) / b, where q = 1 - p
    kelly_fraction = (b * p - (1 - p)) / b

    # 4. Recommended betting strategy
    print(f"\n3. Kelly Criterion Bet Sizing (using true ML odds):")
    print(f"- Empirical win prob p: {p:.3f}")
    print(f"- Avg payout multiplier b: {b:.3f}")
    print(f"- Full Kelly fraction: {kelly_fraction:.3f} of bankroll")
    print(f"- 1/4 Kelly fraction: {kelly_fraction/4:.3f} of bankroll")
    
    print("\n4. Recommended Betting Strategy:")
    print(f"- Only bet when model probability >= {optimal_threshold}")
    print(f"- Expected to bet on approximately {optimal_threshold_row['Coverage (%)']}% of games")
    print(f"- Use fractional Kelly sizing: {kelly_fraction/4:.4f} of bankroll per bet")
    print(f"- Expected ROI: {optimal_threshold_row['ROI (%)']}%")

    
    # 5. Bankroll simulation with different betting strategies
    try:
        print("\n5. BANKROLL SIMULATION")
        
        # Simulate with quarter Kelly betting
        print("\n5.1 Quarter Kelly Betting Strategy:")
        kelly_results = simulate_bankroll(strategy_df, best_thresh, 
                                          initial_bankroll=1000, 
                                          kelly_fraction=0.25)
        
        # Simulate with flat betting (1% of initial bankroll)
        print("\n5.2 Flat Betting Strategy (1% of initial bankroll):")
        flat_results = simulate_bankroll(strategy_df, best_thresh, 
                                         initial_bankroll=1000, 
                                         flat_bet=10)  # $10 = 1% of $1000
        
        # Simulate with flat betting (5% of initial bankroll)
        print("\n5.3 Flat Betting Strategy (5% of initial bankroll):")
        flat_results_5pct = simulate_bankroll(strategy_df, best_thresh, 
                                             initial_bankroll=1000, 
                                             flat_bet=50)  # $50 = 5% of $1000
    except ImportError:
        print("Note: Could not import simulate_bankroll function")
    except Exception as e:
        print(f"Warning: Could not run bankroll simulation: {str(e)}")
        import traceback
        traceback.print_exc()

    return strategy_df, threshold_df, best_thresh

def show_bets_at_threshold(strategy_df, threshold, limit=None):
    """
    Display all bets made at a specific threshold with their moneyline odds and outcomes.
    
    Parameters:
    strategy_df: DataFrame with betting strategy data
    threshold: The probability threshold to filter bets
    limit: Optional limit on number of bets to show (None shows all)
    """
    # Select bets at the specified threshold
    bets = strategy_df.loc[strategy_df['predicted_prob'] >= threshold].copy()
    
    if len(bets) == 0:
        print(f"No bets found at threshold {threshold}")
        return
    
    # Create a view with the essential columns
    bet_view = bets[['moneyLineOdds', 'predicted_prob', 'actual_outcome', 'payout_mult']].copy()
    
    # Add a profit column
    bet_view['profit'] = bet_view.apply(
        lambda r: r['payout_mult'] if r['actual_outcome']==1 else -1.0,
        axis=1
    )
    
    # Sort by predicted probability (descending)
    bet_view = bet_view.sort_values('predicted_prob', ascending=False)
    
    # Add favorite/underdog classification
    bet_view['is_favorite'] = bet_view['moneyLineOdds'] < 0
    bet_view['fav_dog'] = bet_view['is_favorite'].apply(lambda x: 'Favorite' if x else 'Underdog')
    
    # Calculate summary statistics
    total_bets = len(bet_view)
    fav_count = bet_view['is_favorite'].sum()
    dog_count = total_bets - fav_count
    fav_pct = fav_count / total_bets * 100
    dog_pct = dog_count / total_bets * 100
    
    avg_odds = bet_view['moneyLineOdds'].mean()
    median_odds = bet_view['moneyLineOdds'].median()
    
    win_count = bet_view['actual_outcome'].sum()
    win_rate = win_count / total_bets * 100
    
    total_profit = bet_view['profit'].sum()
    roi = total_profit / total_bets * 100
    
    # Print summary statistics
    print(f"\nBets at threshold {threshold}:")
    print(f"Total bets: {total_bets}")
    print(f"Favorites: {fav_count} ({fav_pct:.1f}%), Underdogs: {dog_count} ({dog_pct:.1f}%)")
    print(f"Average ML Odds: {avg_odds:.1f}, Median ML Odds: {median_odds:.1f}")
    print(f"Win rate: {win_rate:.1f}% ({win_count}/{total_bets})")
    print(f"Total profit: {total_profit:.2f} units, ROI: {roi:.1f}%")
    
    # Show detailed bets (limit if specified)
    if limit is not None:
        bet_view = bet_view.head(limit)
        
    # Add the index as a bet number
    bet_view = bet_view.reset_index(drop=True)
    bet_view.index = bet_view.index + 1  # Start from bet #1 instead of 0
    
    # Move the fav_dog column near the front for better readability
    cols = list(bet_view.columns)
    cols.remove('fav_dog')
    cols.insert(1, 'fav_dog')  # Insert after moneyLineOdds
    bet_view = bet_view[cols]
    
    # Print the detailed bets
    print("\nDetailed bets:")
    pd.set_option('display.max_rows', None)  # Ensure all rows are displayed
    print(bet_view)
    
    # Additional stats on winning/losing for favorites and underdogs
    fav_bets = bet_view[bet_view['is_favorite']]
    dog_bets = bet_view[~bet_view['is_favorite']]
    
    if len(fav_bets) > 0:
        fav_win_rate = fav_bets['actual_outcome'].mean() * 100
        fav_profit = fav_bets['profit'].sum()
        fav_roi = fav_profit / len(fav_bets) * 100
        print(f"\nFavorites win rate: {fav_win_rate:.1f}%, ROI: {fav_roi:.1f}%")
    
    if len(dog_bets) > 0:
        dog_win_rate = dog_bets['actual_outcome'].mean() * 100
        dog_profit = dog_bets['profit'].sum()
        dog_roi = dog_profit / len(dog_bets) * 100
        print(f"Underdogs win rate: {dog_win_rate:.1f}%, ROI: {dog_roi:.1f}%")
        
    return bet_view
def simulate_bankroll(strategy_df, threshold, initial_bankroll=1000, kelly_fraction=0.25, flat_bet=None):
    """
    Simulate bankroll growth using either Kelly criterion or flat betting
    
    Parameters:
    strategy_df: DataFrame with betting strategy data
    threshold: The probability threshold to filter bets
    initial_bankroll: Starting bankroll (default $1,000)
    kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)
    flat_bet: If provided, use flat betting of this amount instead of Kelly
    
    Returns:
    DataFrame with bankroll history and bet details
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Select bets at the specified threshold
    bets = strategy_df.loc[strategy_df['predicted_prob'] >= threshold].copy()
    
    if len(bets) == 0:
        print(f"No bets found at threshold {threshold}")
        return None
    
    # Sort chronologically (assuming the dataframe is already in chronological order)
    # If you have a date/game ID column, you would sort by that here
    
    # Create a results dataframe to track bankroll changes
    results = []
    current_bankroll = initial_bankroll
    
    # For each bet
    for i, bet in bets.iterrows():
        # Calculate bet size
        if flat_bet is not None:
            # Flat betting
            bet_size = min(flat_bet, current_bankroll)
        else:
            # Kelly betting
            p = bet['predicted_prob']  # Model's estimated probability
            b = bet['payout_mult']     # Potential profit per $1 wagered
            
            # Kelly formula: f* = (b*p - (1-p)) / b
            full_kelly = (b * p - (1 - p)) / b
            
            # Apply fractional Kelly and ensure we don't bet more than we have
            bet_size = min(current_bankroll, current_bankroll * max(0, full_kelly * kelly_fraction))
            
            # Round to nearest $1 for simplicity
            bet_size = round(bet_size)
        
        # Record pre-bet info
        bet_info = {
            'bankroll_before': current_bankroll,
            'bet_size': bet_size,
            'moneyline_odds': bet['moneyLineOdds'],
            'predicted_prob': bet['predicted_prob'],
            'is_favorite': bet['moneyLineOdds'] < 0
        }
        
        # Calculate outcome
        won = bet['actual_outcome'] == 1
        
        if won:
            profit = bet_size * bet['payout_mult']
            current_bankroll += profit
            bet_info['outcome'] = 'win'
            bet_info['profit'] = profit
        else:
            current_bankroll -= bet_size
            bet_info['outcome'] = 'loss'
            bet_info['profit'] = -bet_size
            
        bet_info['bankroll_after'] = current_bankroll
        
        # Store the results
        results.append(bet_info)
        
        # Check if bankroll is depleted
        if current_bankroll <= 0:
            print("Warning: Bankroll depleted before all bets were placed")
            break
    
    # Convert results to DataFrame
    bankroll_df = pd.DataFrame(results)
    
    # Calculate cumulative stats
    bankroll_df['bet_number'] = range(1, len(bankroll_df) + 1)
    bankroll_df['cumulative_profit'] = bankroll_df['profit'].cumsum()
    bankroll_df['roi'] = (bankroll_df['cumulative_profit'] / 
                         (bankroll_df['bet_number'] * bankroll_df['bet_size'].mean())) * 100
    
    # Overall statistics
    total_bets = len(bankroll_df)
    wins = (bankroll_df['outcome'] == 'win').sum()
    losses = total_bets - wins
    win_rate = wins / total_bets * 100
    final_bankroll = bankroll_df['bankroll_after'].iloc[-1]
    total_profit = final_bankroll - initial_bankroll
    roi = (total_profit / initial_bankroll) * 100
    
    # Report results
    print(f"\n=== BANKROLL SIMULATION RESULTS ===")
    print(f"Initial bankroll: ${initial_bankroll}")
    print(f"Betting strategy: {'Flat ${:.2f} per bet'.format(flat_bet) if flat_bet else f'{kelly_fraction:.2f} Kelly'}")
    print(f"Total bets: {total_bets}")
    print(f"Win rate: {win_rate:.1f}% ({wins} wins, {losses} losses)")
    print(f"Final bankroll: ${final_bankroll:.2f}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.1f}%")
    
    # Visualize bankroll over time
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_df['bet_number'], bankroll_df['bankroll_after'], marker='o', linewidth=2)
    plt.axhline(y=initial_bankroll, color='r', linestyle='--', label=f'Initial ${initial_bankroll}')
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll ($)')
    plt.title('Bankroll Progression')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for big wins/losses
    big_swings = bankroll_df[abs(bankroll_df['profit']) > bankroll_df['profit'].std() * 2]
    for i, swing in big_swings.iterrows():
        plt.annotate(f"${swing['profit']:.0f}",
                    (swing['bet_number'], swing['bankroll_after']),
                    xytext=(0, 10 if swing['profit'] > 0 else -20),
                    textcoords='offset points',
                    ha='center')
    
    plt.savefig('bankroll_simulation.png')
    plt.tight_layout()
    
    # Separate analysis for favorites vs underdogs
    if len(bankroll_df) > 0:
        fav_bets = bankroll_df[bankroll_df['is_favorite']]
        dog_bets = bankroll_df[~bankroll_df['is_favorite']]
        
        print("\nPerformance by odds type:")
        if len(fav_bets) > 0:
            fav_win_rate = (fav_bets['outcome'] == 'win').mean() * 100
            fav_profit = fav_bets['profit'].sum()
            print(f"Favorites: {len(fav_bets)} bets, {fav_win_rate:.1f}% win rate, ${fav_profit:.2f} profit")
        
        if len(dog_bets) > 0:
            dog_win_rate = (dog_bets['outcome'] == 'win').mean() * 100
            dog_profit = dog_bets['profit'].sum()
            print(f"Underdogs: {len(dog_bets)} bets, {dog_win_rate:.1f}% win rate, ${dog_profit:.2f} profit")
    
    return bankroll_df

def main():
    """
    Main execution function for NFL betting analysis pipeline
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv("betting_model_features.csv")
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Perform EDA
        #df = perform_eda(df)
        
        # Preprocess data
        #X_train, X_test, y_train, y_test, features = preprocess_data(df, target_col='this_team_covered')
        X_train, X_test, y_train, y_test, features = preprocess_data(df, target_col = 'api_team_won')
        
        # Build models
        # best_model, results = build_models(X_train, X_test, y_train, y_test, features)
        # after preprocess_data(...)
        best_model, all_models, all_results = build_models(
            X_train, X_test, y_train, y_test, features,
            cv=5,
            tune_hyperparams=True,    # or False
            random_search=True,
            n_iter=20
        )

        
        # Analyze betting strategies
        #strategy_df, optimal_threshold = analyze_betting_strategiesMl(X_test, y_test, best_model)
        strategy_df, thresh_df, best_thresh = analyze_betting_strategiesMl(
            X_test, y_test, best_model
        )
        

        # 4. Show more detailed analysis for the best threshold
        print("\n=== ADDITIONAL ANALYSIS FOR BEST THRESHOLD ===")
        show_bets_at_threshold(strategy_df, best_thresh)
        
        # 5. Run bankroll simulation
        print("\n=== BANKROLL SIMULATION WITH $1,000 STARTING BANKROLL ===")
        # Quarter Kelly
        kelly_results = simulate_bankroll(
            strategy_df, best_thresh, 
            initial_bankroll=1000, 
            kelly_fraction=0.25
        )
        
        # Flat betting 1% and 5% of bankroll
        flat_results_1pct = simulate_bankroll(
            strategy_df, best_thresh, 
            initial_bankroll=1000, 
            flat_bet=10
        )
        
        flat_results_5pct = simulate_bankroll(
            strategy_df, best_thresh, 
            initial_bankroll=1000, 
            flat_bet=50
        )
        print("\nAnalysis complete! Check the visualizations and results.")
        
        return best_model, strategy_df
        
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return None, None
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()








'''
I've examined your NFL betting prediction code and results. This system appears to be analyzing NFL betting data to predict game outcomes and identify profitable betting strategies. Let me break down what I see and provide some thoughts on improvements.

Code Analysis
The code includes a comprehensive pipeline for sports betting analysis:

Exploratory Data Analysis (EDA) - Examining data patterns, missing values, and correlations
Data Preprocessing - Feature selection, handling missing values, chronological train/test split
Model Training - Using Logistic Regression, Random Forest, and Gradient Boosting models
Betting Strategy Analysis - Testing different probability thresholds and Kelly criterion betting
Results Summary
According to the output:

The Logistic Regression model performed best with a ROC AUC of 0.7292
The optimal betting strategy uses a 0.75 probability threshold
At this threshold, the model:
Recommends betting on only 16.7% of games
Projects a 81.4% win rate
Expects an ROI of 55.4%
Suggests using 15.23% of bankroll (quarter Kelly) per bet
Key Observations & Recommendations
The results appear too good to be true
An 81% win rate and 55% ROI in sports betting is extraordinarily high and should be viewed with skepticism
Sports betting markets are generally efficient; sustained performance above 55% is rare
Potential Data Leakage Issues
The target variable was changed from this_team_covered to api_team_won
There may be features inadvertently leaking information about game outcomes
I notice moneyLineOdds is the most important feature in two models - these odds directly reflect win probability
Chronological Validation Concerns
While the code uses TimeSeriesSplit for cross-validation, it's crucial to ensure no future information leaks into training
Overfitting Risk
The high win rate at the 0.75 threshold but on only 16.7% of games suggests potential overfitting
The model might be finding patterns that don't generalize to new data
Suggested Improvements
Feature Engineering
Create aggregated team performance metrics (rolling averages, momentum indicators)
Add features about team matchups, historical head-to-head records
Consider weather data, injuries, and other external factors
Enhanced Validation
Implement walk-forward analysis with expanding window validation
Test on multiple seasons not seen during training
Calculate confidence intervals for win rates and ROI
Ensemble Approach
Combine predictions from multiple models for more robust forecasting
Consider model stacking with a meta-learner
Additional Analysis
Analyze performance by bet type (spread, moneyline, over/under)
Test for consistent performance across different seasons/years
Evaluate against specific teams, home/away scenarios
Threshold Selection
The current threshold of 0.75 is extremely selective
Test multiple thresholds on a separate validation set before finalizing
Bankroll Management
Quarter Kelly (15.23%) is still quite aggressive for sports betting
Consider flatter betting strategies (1-5% per bet) for risk management
Next Steps
Review the dataset to identify and remove potential sources of data leakage
Re-run the analysis with stricter chronological validation
Implement additional features that might improve predictive power
Test on more recent seasons as out-of-sample validation
Consider a more conservative betting approach than the recommended strategy

'''
