import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



def plot_linear_prediction_waterfall(index, linear_components, model, X_test_selected):
    """
    Create a waterfall chart showing how a linear model prediction is built from the intercept
    and individual feature contributions for a specific data point, with features sorted by
    the magnitude of their contributions.
    
    Parameters:
    -----------
    index : int
        Index of the data point to explain
    linear_components : DataFrame
        DataFrame containing the linear model components (intercept and feature contributions)
    model : LinearRegression
        The trained linear model
    X_test_selected : DataFrame
        Test data with selected features
    """
    import matplotlib.pyplot as plt
    
    # Get components for the specified index
    components = linear_components.iloc[index].copy()
    
    # Extract intercept and feature contributions
    intercept = components['Intercept']
    feature_contributions = components.drop('Intercept')
    
    # Sort feature contributions by absolute magnitude
    sorted_contributions = feature_contributions.reindex(
        feature_contributions.abs().sort_values(ascending=False).index
    )
    
    # Prepare data for the waterfall chart
    features = ['Intercept'] + sorted_contributions.index.tolist()
    values = [intercept] + sorted_contributions.values.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Starting point (intercept)
    cumulative = values[0]
    ax.bar(0, values[0], bottom=0, color='lightgray', label='Intercept')
    
    # Add feature contributions
    for i in range(1, len(features)):
        if values[i] > 0:
            color = 'lightblue'
        else:
            color = 'red'
        ax.bar(i, values[i], bottom=cumulative, color=color, alpha=0.7)
        cumulative += values[i]
    
    # Add final prediction point
    ax.bar(len(features), 0, bottom=cumulative, color='blue', label='Final prediction')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add horizontal line at prediction
    ax.axhline(y=cumulative, color='blue', linestyle='--', alpha=0.7, 
              label=f'Prediction: {cumulative:.3f}')
    
    # Set x-axis labels
    plt.xticks(range(len(features) + 1), features + ['Prediction'])
    plt.xticks(rotation=45, ha='right')
    
    # Set title and labels
    plt.title(f'Linear Model Prediction Breakdown for Index {index}')
    plt.ylabel('Contribution to Prediction')
    
    # Extract the original feature values to display
    original_values = X_test_selected.iloc[index]
    feature_text = "\n".join([f"{feat}: {original_values[feat]:.3f}" 
                             for feat in original_values.index])
    
    # Add feature values as text box in the top-left, offset from the legend
    plt.figtext(0.02, 0.85, f"Feature values:\n{feature_text}", 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend with offset to avoid overlap with the text box
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.75))
    
    # Add padding and adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Show the plot
    plt.show()
    
    # Print prediction summary
    print(f"Prediction: {cumulative:.3f}")
    
    # Show sorted contributions
    print("\nFeature contributions (sorted by magnitude):")
    for feat, value in sorted_contributions.abs().sort_values(ascending=False).items():
        original_value = sorted_contributions[feat]
        print(f"{feat}: {original_value:.3f}")




def plot_permuted_correlated_features():

    np.random.seed(42)

    n_samples = 1000
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)

    df_sample = pd.DataFrame({'feature1': x1, 'feature2': x2})

    X_permuted = df_sample.copy()

    # Select a slice from the middle of feature1's range to permute
    lower_bound = np.percentile(df_sample['feature1'], 25)
    upper_bound = np.percentile(df_sample['feature1'], 75)
    mask = (X_permuted['feature1'] >= lower_bound) & (X_permuted['feature1'] <= upper_bound)
    indices_to_permute = X_permuted[mask].index

    # Save original values for visualization
    original_values = X_permuted.loc[indices_to_permute, 'feature1'].copy()

    # Permute feature1 values within the selected range
    X_permuted.loc[indices_to_permute, 'feature1'] = np.random.permutation(original_values)

    # Calculate correlation
    original_corr = df_sample['feature1'].corr(df_sample['feature2'])
    permuted_corr = X_permuted['feature1'].corr(X_permuted['feature2'])

    # Create scatter plot to visualize the effect of permutation
    plt.figure(figsize=(8, 8))

    # Plot all points in the original data
    plt.scatter(df_sample['feature1'], df_sample['feature2'], 
                alpha=0.5, label='Original Points', color='blue')

    # Highlight the permuted points
    plt.scatter(X_permuted.loc[indices_to_permute, 'feature1'], 
                X_permuted.loc[indices_to_permute, 'feature2'], 
                alpha=0.7, label='Permuted Points', color='red')

    # Add reference lines for the slice boundaries
    plt.axvline(x=lower_bound, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=upper_bound, color='gray', linestyle='--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Demonstration of Permutation Method Issues with Correlated Features\n'
            f'Original Correlation: {original_corr:.3f}')
    plt.legend()
    plt.grid(alpha=0.3)

    # Add annotation explaining the problem
    plt.annotate(
        "Permutation creates\npoints outside the\njoint distribution",
        xy=(-0.5, 1.5), xytext=(0.5, 2.0),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def map_point(lat, lon, price, predicted_price, median_income, house_age, ave_rooms, ave_bedrooms, population, ave_occup):
    # Create a map centered at the sample point
    m = folium.Map(location=[lat, lon], zoom_start=10)

    # Add a marker for the sample point
    folium.Marker(
        location=[lat, lon],
        popup=f"Price: ${price:.3f}M<br>Predicted: ${predicted_price:.3f}M<br>"
              f"Median Income: {median_income}<br>House Age: {house_age}<br>"
              f"Avg Rooms: {ave_rooms}<br>Avg Bedrooms: {ave_bedrooms}<br>"
              f"Population: {population}<br>Avg Occupancy: {ave_occup}",
        icon=folium.Icon(color="red")
    ).add_to(m)
    return m




def train_rf_model(X_train, y_train, X_test, y_test):


    # Set up hyperparameter search for Random Forest Regressor
    # Define the parameter grid to search
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

# Create the random forest regressor
    rf = RandomForestRegressor(random_state=42)

    # Set up k-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create the RandomizedSearchCV
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings to try
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Fit the random search
    rf_random.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", rf_random.best_params_)
    print("Best CV Score:", -rf_random.best_score_)  # Convert back to positive MSE

    # Save best parameters to JSON file

    best_params = rf_random.best_params_

    print("Best parameters saved to rf_best_params.json")

    # Evaluate on test set
    best_rf = rf_random.best_estimator_
    test_score = best_rf.score(X_test, y_test)
    print(f"R² Score on Test Set: {test_score:.4f}")

    # Calculate MSE on test set
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE on Test Set: {mse:.4f}")

    return best_rf, best_params