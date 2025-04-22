import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
    fig, ax = plt.subplots(figsize=(12, 8))
    
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