import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_churn_distribution(data):
    """Plot churn distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    churn_counts = data['Churn'].value_counts()
    ax1 = sns.countplot(x='Churn', data=data, ax=axes[0])
    axes[0].set_title('Churn Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(churn_counts):
        axes[0].text(i, v, str(v), ha='center', va='bottom')
    
    # Pie chart
    colors = ['#66b3ff', '#ff9999']
    axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Churn Percentage')
    
    plt.tight_layout()
    plt.show()

def plot_tenure_vs_churn(data):
    """Plot tenure distribution by churn"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=data)
    plt.title('Tenure Distribution by Churn Status')
    plt.ylabel('Tenure (months)')
    plt.show()

def plot_contract_type_analysis(data):
    """Analyze churn by contract type"""
    contract_churn = pd.crosstab(data['Contract'], data['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', figsize=(10, 6), color=['#66b3ff', '#ff9999'])
    plt.title('Churn Rate by Contract Type')
    plt.ylabel('Percentage')
    plt.xlabel('Contract Type')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_service_usage(data):
    """Analyze churn by service usage"""
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    fig = make_subplots(rows=3, cols=3, subplot_titles=services)
    
    for i, service in enumerate(services):
        row = i // 3 + 1
        col = i % 3 + 1
        temp = data.groupby([service, 'Churn']).size().unstack(fill_value=0)
        temp_pct = temp.div(temp.sum(axis=1), axis=0) * 100
        
        fig.add_trace(
            go.Bar(x=temp_pct.index, y=temp_pct[1], name='Churn',
                   marker_color='#ff9999', showlegend=(i==0)),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(x=temp_pct.index, y=temp_pct[0], name='No Churn',
                   marker_color='#66b3ff', showlegend=(i==0)),
            row=row, col=col
        )
    
    fig.update_layout(height=800, title_text="Churn Rate by Service")
    fig.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix"""
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = data[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importance attribute")
        return
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance