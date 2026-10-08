"""
CodeAlpha Task 3 - Data Visualization Project
Professional Sales Data Analysis and Visualization
Author: CodeAlpha Intern
Date: October 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('default')
sns.set_palette("husl")

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['dataset', 'charts', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print("✓ Project directories verified")

def load_and_validate_data():
    """Load dataset and perform initial validation"""
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset/task3_dataset.csv')
        print(f"✓ Dataset loaded successfully")
        print(f"  - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("❌ Dataset not found!")
        return None

def display_data_overview(df):
    """Display comprehensive data overview"""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found ✓")
    else:
        print(missing[missing > 0])
    
    print(f"\nDuplicate Records: {df.duplicated().sum()}")
    
    print(f"\nNumerical Columns:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  {numerical_cols}")
    
    print(f"\nCategorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"  {categorical_cols}")
    
    return numerical_cols, categorical_cols

def prepare_data(df):
    """Prepare and clean dataset for analysis"""
    print("\nPreparing dataset...")
    
    # Create a copy for cleaning
    cleaned_df = df.copy()
    
    # Convert date column to datetime
    if 'order_date' in cleaned_df.columns:
        cleaned_df['order_date'] = pd.to_datetime(cleaned_df['order_date'])
        print("✓ Date column converted to datetime")
    
    # Remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    if removed_duplicates > 0:
        print(f"✓ Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values (if any)
    missing_before = cleaned_df.isnull().sum().sum()
    cleaned_df = cleaned_df.dropna()
    missing_after = missing_before - cleaned_df.isnull().sum().sum()
    if missing_after > 0:
        print(f"✓ Handled {missing_after} missing values")
    
    # Add calculated columns for analysis
    if 'total_amount' in cleaned_df.columns and 'profit_margin' in cleaned_df.columns:
        cleaned_df['profit_amount'] = cleaned_df['total_amount'] * cleaned_df['profit_margin']
    
    if 'order_date' in cleaned_df.columns:
        cleaned_df['year'] = cleaned_df['order_date'].dt.year
        cleaned_df['month'] = cleaned_df['order_date'].dt.month
        cleaned_df['month_name'] = cleaned_df['order_date'].dt.month_name()
    
    print(f"✓ Data preparation completed")
    print(f"  Final dataset: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    
    # Save cleaned dataset
    cleaned_df.to_csv('output/cleaned_dataset.csv', index=False)
    print("✓ Cleaned dataset saved to output/cleaned_dataset.csv")
    
    return cleaned_df

def calculate_kpis(df):
    """Calculate Key Performance Indicators"""
    print("\nCalculating KPIs...")
    
    kpis = {
        'total_records': len(df),
        'total_customers': df['customer_id'].nunique(),
        'total_revenue': df['total_amount'].sum(),
        'average_order_value': df['total_amount'].mean(),
        'max_order_value': df['total_amount'].max(),
        'min_order_value': df['total_amount'].min(),
        'total_categories': df['product_category'].nunique(),
        'top_category': df.groupby('product_category')['total_amount'].sum().idxmax(),
        'top_region': df.groupby('region')['total_amount'].sum().idxmax(),
        'top_salesperson': df.groupby('salesperson')['total_amount'].sum().idxmax(),
    }
    
    if 'profit_amount' in df.columns:
        kpis['total_profit'] = df['profit_amount'].sum()
        kpis['average_profit_margin'] = df['profit_margin'].mean()
    
    print("✓ KPIs calculated successfully")
    return kpis
def create_visualization_1_distribution(df):
    """Create distribution analysis of total amounts"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['total_amount'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Order Values', fontsize=14, fontweight='bold')
    plt.xlabel('Total Amount ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['total_amount'], color='lightgreen')
    plt.title('Order Value Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/01_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 1: Distribution Analysis completed")

def create_visualization_2_category_comparison(df):
    """Create category comparison analysis"""
    plt.figure(figsize=(14, 8))
    
    category_sales = df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(category_sales.index, category_sales.values, color='steelblue', alpha=0.8)
    plt.title('Total Revenue by Product Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(2, 1, 2)
    category_counts = df['product_category'].value_counts()
    plt.bar(category_counts.index, category_counts.values, color='coral', alpha=0.8)
    plt.title('Number of Orders by Product Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('charts/02_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 2: Category Comparison completed")

def create_visualization_3_trend_analysis(df):
    """Create time series trend analysis"""
    if 'order_date' not in df.columns:
        print("❌ No date column found for trend analysis")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Monthly revenue trend
    monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['total_amount'].sum()
    
    plt.subplot(2, 2, 1)
    monthly_revenue.plot(kind='line', marker='o', linewidth=2, markersize=8, color='darkblue')
    plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Daily order count
    daily_orders = df.groupby(df['order_date'].dt.date).size()
    
    plt.subplot(2, 2, 2)
    daily_orders.plot(kind='line', alpha=0.7, color='green')
    plt.title('Daily Order Count Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Monthly orders by category
    plt.subplot(2, 1, 2)
    monthly_category = df.groupby([df['order_date'].dt.to_period('M'), 'product_category'])['total_amount'].sum().unstack(fill_value=0)
    monthly_category.plot(kind='area', stacked=True, alpha=0.7)
    plt.title('Monthly Revenue by Category (Stacked Area)', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/03_trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 3: Trend Analysis completed")
def create_visualization_4_correlation_heatmap(df):
    """Create correlation analysis heatmap"""
    plt.figure(figsize=(12, 8))
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('charts/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 4: Correlation Heatmap completed")

def create_visualization_5_scatter_relationship(df):
    """Create scatter plot analysis"""
    plt.figure(figsize=(15, 10))
    
    # Scatter plot 1: Quantity vs Total Amount
    plt.subplot(2, 2, 1)
    plt.scatter(df['quantity'], df['total_amount'], alpha=0.6, color='purple', s=50)
    plt.title('Quantity vs Total Amount', fontsize=14, fontweight='bold')
    plt.xlabel('Quantity', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Scatter plot 2: Unit Price vs Total Amount (colored by category)
    plt.subplot(2, 2, 2)
    categories = df['product_category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    for i, category in enumerate(categories):
        cat_data = df[df['product_category'] == category]
        plt.scatter(cat_data['unit_price'], cat_data['total_amount'], 
                   alpha=0.6, color=colors[i], label=category, s=50)
    plt.title('Unit Price vs Total Amount by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Unit Price ($)', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot 3: Customer Age vs Order Value
    plt.subplot(2, 2, 3)
    colors_gender = {'M': 'blue', 'F': 'red'}
    for gender in df['customer_gender'].unique():
        gender_data = df[df['customer_gender'] == gender]
        plt.scatter(gender_data['customer_age'], gender_data['total_amount'],
                   alpha=0.6, color=colors_gender[gender], label=f'Gender {gender}', s=50)
    plt.title('Customer Age vs Order Value by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Age', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot 4: Profit Margin vs Total Amount
    plt.subplot(2, 2, 4)
    plt.scatter(df['profit_margin'], df['total_amount'], alpha=0.6, color='green', s=50)
    plt.title('Profit Margin vs Total Amount', fontsize=14, fontweight='bold')
    plt.xlabel('Profit Margin', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/05_scatter_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 5: Scatter Relationship Analysis completed")

def create_visualization_6_boxplot_outliers(df):
    """Create box plot analysis for outlier detection"""
    plt.figure(figsize=(16, 12))
    
    # Box plot 1: Total Amount by Category
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='product_category', y='total_amount', palette='Set2')
    plt.title('Order Value Distribution by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Box plot 2: Total Amount by Region
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='region', y='total_amount', palette='Set1')
    plt.title('Order Value Distribution by Region', fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Box plot 3: Customer Age by Gender
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='customer_gender', y='customer_age', palette='pastel')
    plt.title('Customer Age Distribution by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Customer Age', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Box plot 4: Profit Margin by Payment Method
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='payment_method', y='profit_margin', palette='husl')
    plt.title('Profit Margin Distribution by Payment Method', fontsize=14, fontweight='bold')
    plt.xlabel('Payment Method', fontsize=12)
    plt.ylabel('Profit Margin', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('charts/06_boxplot_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 6: Box Plot Outlier Analysis completed")
def create_visualization_7_ranked_categories(df):
    """Create ranked analysis charts"""
    plt.figure(figsize=(16, 10))
    
    # Top products by revenue
    plt.subplot(2, 2, 1)
    product_revenue = df.groupby('product_name')['total_amount'].sum().sort_values(ascending=True).tail(10)
    plt.barh(range(len(product_revenue)), product_revenue.values, color='teal', alpha=0.8)
    plt.yticks(range(len(product_revenue)), product_revenue.index)
    plt.title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
    plt.xlabel('Total Revenue ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Top customers by spending
    plt.subplot(2, 2, 2)
    customer_spending = df.groupby('customer_id')['total_amount'].sum().sort_values(ascending=True).tail(10)
    plt.barh(range(len(customer_spending)), customer_spending.values, color='orange', alpha=0.8)
    plt.yticks(range(len(customer_spending)), customer_spending.index)
    plt.title('Top 10 Customers by Spending', fontsize=14, fontweight='bold')
    plt.xlabel('Total Spending ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Salesperson performance
    plt.subplot(2, 2, 3)
    salesperson_performance = df.groupby('salesperson')['total_amount'].sum().sort_values(ascending=False)
    bars = plt.bar(salesperson_performance.index, salesperson_performance.values, color='purple', alpha=0.8)
    plt.title('Salesperson Performance by Revenue', fontsize=14, fontweight='bold')
    plt.xlabel('Salesperson', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Regional performance
    plt.subplot(2, 2, 4)
    regional_performance = df.groupby('region')['total_amount'].sum().sort_values(ascending=False)
    colors = ['gold', 'silver', '#CD7F32', 'lightblue']  # Gold, Silver, Bronze, Light Blue
    bars = plt.bar(regional_performance.index, regional_performance.values, color=colors, alpha=0.8)
    plt.title('Regional Sales Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/07_ranked_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 7: Ranked Categories Analysis completed")

def create_visualization_8_composition_analysis(df):
    """Create composition analysis charts"""
    plt.figure(figsize=(16, 8))
    
    # Payment method distribution
    plt.subplot(1, 2, 1)
    payment_counts = df['payment_method'].value_counts()
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(payment_counts)))
    wedges, texts, autotexts = plt.pie(payment_counts.values, labels=payment_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Payment Method Distribution', fontsize=16, fontweight='bold')
    
    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Gender distribution with revenue
    plt.subplot(1, 2, 2)
    gender_revenue = df.groupby('customer_gender')['total_amount'].sum()
    colors_gender = ['lightpink', 'lightblue']
    wedges, texts, autotexts = plt.pie(gender_revenue.values, labels=['Female', 'Male'], 
                                      autopct='%1.1f%%', colors=colors_gender, startangle=90)
    plt.title('Revenue Distribution by Gender', fontsize=16, fontweight='bold')
    
    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig('charts/08_composition_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 8: Composition Analysis completed")

def create_visualization_9_multivariate_analysis(df):
    """Create multivariate analysis"""
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Revenue by Region and Category (Heatmap)
    plt.subplot(2, 2, 1)
    pivot_data = df.pivot_table(values='total_amount', index='region', 
                               columns='product_category', aggfunc='sum', fill_value=0)
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'shrink': 0.8})
    plt.title('Revenue Heatmap: Region × Category', fontsize=14, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    
    # Subplot 2: Age groups vs Category spending
    plt.subplot(2, 2, 2)
    df['age_group'] = pd.cut(df['customer_age'], bins=[20, 30, 40, 50], labels=['20-30', '30-40', '40-50'])
    age_category = df.groupby(['age_group', 'product_category'])['total_amount'].sum().unstack(fill_value=0)
    age_category.plot(kind='bar', stacked=True, alpha=0.8, ax=plt.gca())
    plt.title('Spending by Age Group and Category', fontsize=14, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Subplot 3: Bubble chart (Customer Age vs Order Value vs Quantity)
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(df['customer_age'], df['total_amount'], s=df['quantity']*20, 
                         c=df['profit_margin'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Profit Margin')
    plt.title('Age vs Order Value (Bubble Size = Quantity)', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Age', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Monthly sales by payment method
    plt.subplot(2, 2, 4)
    if 'month_name' in df.columns:
        monthly_payment = df.groupby(['month_name', 'payment_method'])['total_amount'].sum().unstack(fill_value=0)
        monthly_payment.plot(kind='bar', alpha=0.8, ax=plt.gca())
        plt.title('Monthly Sales by Payment Method', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Amount ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('charts/09_multivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 9: Multivariate Analysis completed")
def create_visualization_10_final_dashboard(df, kpis):
    """Create comprehensive final dashboard"""
    fig = plt.figure(figsize=(20, 14))
    
    # Set up the dashboard layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Dashboard title
    fig.suptitle('CodeAlpha Task 3 - Sales Performance Dashboard', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # KPI Cards (Top row)
    kpi_data = [
        ('Total Revenue', f"${kpis['total_revenue']:,.0f}", 'green'),
        ('Total Orders', f"{kpis['total_records']:,}", 'blue'),
        ('Avg Order Value', f"${kpis['average_order_value']:,.0f}", 'orange'),
        ('Total Customers', f"{kpis['total_customers']:,}", 'purple')
    ]
    
    for i, (title, value, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.7, value, ha='center', va='center', fontsize=20, 
               fontweight='bold', transform=ax.transAxes, color=color)
        ax.text(0.5, 0.3, title, ha='center', va='center', fontsize=12, 
               fontweight='bold', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color(color)
    
    # Chart 1: Category Performance
    ax1 = fig.add_subplot(gs[1, :2])
    category_revenue = df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
    bars = ax1.bar(category_revenue.index, category_revenue.values, color='steelblue', alpha=0.8)
    ax1.set_title('Revenue by Product Category', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chart 2: Regional Performance
    ax2 = fig.add_subplot(gs[1, 2:])
    regional_data = df.groupby('region')['total_amount'].sum().sort_values(ascending=False)
    colors = ['gold', 'silver', '#CD7F32', 'lightblue']
    bars = ax2.bar(regional_data.index, regional_data.values, color=colors[:len(regional_data)], alpha=0.8)
    ax2.set_title('Revenue by Region', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Revenue ($)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 3: Monthly Trend
    ax3 = fig.add_subplot(gs[2, :2])
    if 'order_date' in df.columns:
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['total_amount'].sum()
        monthly_revenue.plot(kind='line', marker='o', linewidth=3, markersize=8, 
                           color='darkgreen', ax=ax3)
        ax3.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Revenue ($)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # Chart 4: Top Performers
    ax4 = fig.add_subplot(gs[2, 2:])
    salesperson_performance = df.groupby('salesperson')['total_amount'].sum().sort_values(ascending=False)
    bars = ax4.bar(salesperson_performance.index, salesperson_performance.values, 
                  color='purple', alpha=0.8)
    ax4.set_title('Salesperson Performance', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Revenue ($)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Summary insights box
    ax5 = fig.add_subplot(gs[3, :])
    insights_text = f"""
    KEY INSIGHTS & RECOMMENDATIONS:
    
    🏆 TOP CATEGORY: {kpis['top_category']} generates the highest revenue
    🌍 TOP REGION: {kpis['top_region']} leads in regional performance  
    👤 TOP SALESPERSON: {kpis['top_salesperson']} achieves highest sales
    💰 AVERAGE ORDER: ${kpis['average_order_value']:,.0f} per transaction
    📊 TOTAL PROFIT: ${kpis.get('total_profit', 0):,.0f} across all orders
    
    🎯 BUSINESS RECOMMENDATIONS:
    • Focus marketing efforts on {kpis['top_category']} category
    • Expand operations in {kpis['top_region']} region
    • Replicate {kpis['top_salesperson']}'s sales strategies across the team
    • Target customer segments with order values above ${kpis['average_order_value']:,.0f}
    """
    
    ax5.text(0.02, 0.95, insights_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax5.axis('off')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
             fontsize=10, style='italic', alpha=0.7)
    
    plt.savefig('charts/10_final_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 10: Final Dashboard completed")
def generate_insights(df, kpis):
    """Generate comprehensive data insights"""
    print("\nGenerating insights...")
    
    insights = []
    insights.append("CODEALPHA TASK 3 — DATA VISUALIZATION INSIGHTS")
    insights.append("=" * 60)
    insights.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    insights.append("")
    
    # Dataset Overview
    insights.append("1. DATASET OVERVIEW")
    insights.append("-" * 30)
    insights.append(f"• Total Records: {kpis['total_records']:,}")
    insights.append(f"• Dataset Coverage: {df['order_date'].min().strftime('%B %Y')} to {df['order_date'].max().strftime('%B %Y')}")
    insights.append(f"• Unique Customers: {kpis['total_customers']:,}")
    insights.append(f"• Product Categories: {kpis['total_categories']}")
    insights.append("")
    
    # Revenue Insights
    insights.append("2. REVENUE ANALYSIS")
    insights.append("-" * 30)
    insights.append(f"• Total Revenue: ${kpis['total_revenue']:,.2f}")
    insights.append(f"• Average Order Value: ${kpis['average_order_value']:,.2f}")
    insights.append(f"• Highest Single Order: ${kpis['max_order_value']:,.2f}")
    insights.append(f"• Lowest Single Order: ${kpis['min_order_value']:,.2f}")
    if 'total_profit' in kpis:
        insights.append(f"• Total Profit: ${kpis['total_profit']:,.2f}")
        insights.append(f"• Average Profit Margin: {kpis['average_profit_margin']:.1%}")
    insights.append("")
    
    # Category Insights
    category_analysis = df.groupby('product_category').agg({
        'total_amount': ['sum', 'count', 'mean'],
        'quantity': 'sum'
    }).round(2)
    
    insights.append("3. PRODUCT CATEGORY INSIGHTS")
    insights.append("-" * 30)
    top_category_revenue = df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
    insights.append(f"• Highest Revenue Category: {top_category_revenue.index[0]} (${top_category_revenue.iloc[0]:,.0f})")
    insights.append(f"• Lowest Revenue Category: {top_category_revenue.index[-1]} (${top_category_revenue.iloc[-1]:,.0f})")
    
    category_orders = df['product_category'].value_counts()
    insights.append(f"• Most Ordered Category: {category_orders.index[0]} ({category_orders.iloc[0]} orders)")
    insights.append(f"• Revenue Distribution: {top_category_revenue.index[0]} accounts for {(top_category_revenue.iloc[0]/kpis['total_revenue']*100):.1f}% of total revenue")
    insights.append("")
    
    # Regional Insights
    insights.append("4. REGIONAL PERFORMANCE")
    insights.append("-" * 30)
    regional_performance = df.groupby('region')['total_amount'].sum().sort_values(ascending=False)
    insights.append(f"• Top Performing Region: {regional_performance.index[0]} (${regional_performance.iloc[0]:,.0f})")
    insights.append(f"• Lowest Performing Region: {regional_performance.index[-1]} (${regional_performance.iloc[-1]:,.0f})")
    
    regional_orders = df.groupby('region').size().sort_values(ascending=False)
    insights.append(f"• Most Active Region: {regional_orders.index[0]} ({regional_orders.iloc[0]} orders)")
    insights.append(f"• Regional Revenue Gap: {((regional_performance.iloc[0] - regional_performance.iloc[-1])/regional_performance.iloc[0]*100):.1f}% difference between top and bottom regions")
    insights.append("")
    
    # Customer Demographics
    insights.append("5. CUSTOMER DEMOGRAPHICS")
    insights.append("-" * 30)
    avg_age_by_gender = df.groupby('customer_gender')['customer_age'].mean()
    insights.append(f"• Average Customer Age - Male: {avg_age_by_gender['M']:.1f} years")
    insights.append(f"• Average Customer Age - Female: {avg_age_by_gender['F']:.1f} years")
    
    gender_spending = df.groupby('customer_gender')['total_amount'].sum()
    total_spending = gender_spending.sum()
    insights.append(f"• Male Customer Revenue: ${gender_spending['M']:,.0f} ({gender_spending['M']/total_spending*100:.1f}%)")
    insights.append(f"• Female Customer Revenue: ${gender_spending['F']:,.0f} ({gender_spending['F']/total_spending*100:.1f}%)")
    
    age_ranges = pd.cut(df['customer_age'], bins=[20, 30, 40, 50], labels=['20-30', '30-40', '40-50'])
    age_spending = df.groupby(age_ranges)['total_amount'].sum().sort_values(ascending=False)
    insights.append(f"• Highest Spending Age Group: {age_spending.index[0]} (${age_spending.iloc[0]:,.0f})")
    insights.append("")
    
    # Salesperson Performance
    insights.append("6. SALESPERSON PERFORMANCE")
    insights.append("-" * 30)
    salesperson_stats = df.groupby('salesperson').agg({
        'total_amount': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    top_salesperson = df.groupby('salesperson')['total_amount'].sum().sort_values(ascending=False)
    insights.append(f"• Top Performer: {top_salesperson.index[0]} (${top_salesperson.iloc[0]:,.0f})")
    insights.append(f"• Performance Gap: {((top_salesperson.iloc[0] - top_salesperson.iloc[-1])/top_salesperson.iloc[0]*100):.1f}% difference between top and bottom performer")
    
    salesperson_orders = df.groupby('salesperson').size().sort_values(ascending=False)
    insights.append(f"• Most Active Salesperson: {salesperson_orders.index[0]} ({salesperson_orders.iloc[0]} orders)")
    insights.append("")
    
    # Payment Method Analysis
    insights.append("7. PAYMENT METHOD ANALYSIS")
    insights.append("-" * 30)
    payment_revenue = df.groupby('payment_method')['total_amount'].sum().sort_values(ascending=False)
    payment_counts = df['payment_method'].value_counts()
    
    insights.append(f"• Preferred Payment Method: {payment_counts.index[0]} ({payment_counts.iloc[0]} transactions)")
    insights.append(f"• Highest Revenue Payment Method: {payment_revenue.index[0]} (${payment_revenue.iloc[0]:,.0f})")
    
    avg_order_by_payment = df.groupby('payment_method')['total_amount'].mean().sort_values(ascending=False)
    insights.append(f"• Highest Average Order Value: {avg_order_by_payment.index[0]} (${avg_order_by_payment.iloc[0]:,.0f} per order)")
    insights.append("")
    
    # Time-based Insights
    if 'order_date' in df.columns:
        insights.append("8. TEMPORAL INSIGHTS")
        insights.append("-" * 30)
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['total_amount'].sum()
        insights.append(f"• Best Month: {monthly_revenue.idxmax()} (${monthly_revenue.max():,.0f})")
        insights.append(f"• Lowest Month: {monthly_revenue.idxmin()} (${monthly_revenue.min():,.0f})")
        
        monthly_growth = monthly_revenue.pct_change().dropna()
        if len(monthly_growth) > 0:
            avg_growth = monthly_growth.mean()
            insights.append(f"• Average Monthly Growth: {avg_growth:.1%}")
        insights.append("")
    
    # Key Correlations
    insights.append("9. KEY RELATIONSHIPS")
    insights.append("-" * 30)
    correlation_matrix = df[['quantity', 'unit_price', 'total_amount', 'customer_age', 'profit_margin']].corr()
    
    # Find strongest correlations (excluding self-correlations)
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    insights.append(f"• Strongest Positive Correlation: {correlations[0][0]} ↔ {correlations[0][1]} (r = {correlations[0][2]:.3f})")
    for corr in correlations:
        if corr[2] < 0:
            insights.append(f"• Strongest Negative Correlation: {corr[0]} ↔ {corr[1]} (r = {corr[2]:.3f})")
            break
    insights.append("")
    
    # Business Recommendations
    insights.append("10. DECISION-MAKING INSIGHTS & RECOMMENDATIONS")
    insights.append("-" * 30)
    insights.append("STRATEGIC RECOMMENDATIONS:")
    insights.append(f"• FOCUS AREA: Invest more resources in {kpis['top_category']} category (highest revenue generator)")
    insights.append(f"• EXPANSION: Consider expanding operations in {kpis['top_region']} region")
    insights.append(f"• TRAINING: Study and replicate {kpis['top_salesperson']}'s sales strategies")
    insights.append(f"• CUSTOMER TARGETING: Focus on customers with order values above ${kpis['average_order_value']:,.0f}")
    
    # Category-specific recommendations
    category_margins = df.groupby('product_category')['profit_margin'].mean().sort_values(ascending=False)
    insights.append(f"• PROFIT OPTIMIZATION: {category_margins.index[0]} has the highest profit margins ({category_margins.iloc[0]:.1%})")
    
    insights.append("")
    insights.append("OPERATIONAL RECOMMENDATIONS:")
    insights.append(f"• Payment Processing: Optimize {payment_counts.index[0]} payment processing (most used method)")
    
    low_performing_region = regional_performance.index[-1]
    insights.append(f"• Regional Support: Provide additional support to {low_performing_region} region")
    
    insights.append(f"• Customer Retention: Target {age_spending.index[0]} age group for loyalty programs")
    
    insights.append("")
    insights.append("11. FINAL CONCLUSION")
    insights.append("-" * 30)
    insights.append(f"The analysis of {kpis['total_records']:,} sales records reveals strong performance in {kpis['top_category']}")
    insights.append(f"category and {kpis['top_region']} region. With total revenue of ${kpis['total_revenue']:,.0f} and average")
    insights.append(f"order value of ${kpis['average_order_value']:,.0f}, the business shows healthy metrics.")
    insights.append(f"Key opportunities exist in optimizing underperforming regions and categories.")
    insights.append("")
    insights.append("This analysis supports data-driven decision making for:")
    insights.append("• Resource allocation and investment priorities")
    insights.append("• Regional expansion and optimization strategies") 
    insights.append("• Sales team performance improvement")
    insights.append("• Customer segmentation and targeting")
    insights.append("• Product portfolio optimization")
    
    # Save insights to file
    with open('output/insights.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(insights))
    
    print("✓ Comprehensive insights generated and saved")
    return insights

def main():
    """Main execution function"""
    print("🚀 STARTING CODEALPHA TASK 3 - DATA VISUALIZATION PROJECT")
    print("=" * 70)
    
    # Create necessary directories
    create_directories()
    
    # Load and validate data
    df = load_and_validate_data()
    if df is None:
        print("❌ Failed to load dataset. Exiting...")
        return
    
    # Display data overview
    numerical_cols, categorical_cols = display_data_overview(df)
    
    # Prepare data
    cleaned_df = prepare_data(df)
    
    # Calculate KPIs
    kpis = calculate_kpis(cleaned_df)
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create all visualizations
    create_visualization_1_distribution(cleaned_df)
    create_visualization_2_category_comparison(cleaned_df)
    create_visualization_3_trend_analysis(cleaned_df)
    create_visualization_4_correlation_heatmap(cleaned_df)
    create_visualization_5_scatter_relationship(cleaned_df)
    create_visualization_6_boxplot_outliers(cleaned_df)
    create_visualization_7_ranked_categories(cleaned_df)
    create_visualization_8_composition_analysis(cleaned_df)
    create_visualization_9_multivariate_analysis(cleaned_df)
    create_visualization_10_final_dashboard(cleaned_df, kpis)
    
    # Generate insights
    insights = generate_insights(cleaned_df, kpis)
    
    print(f"\n{'='*60}")
    print("PROJECT COMPLETION SUMMARY")
    print("="*60)
    print("✅ All visualizations created successfully")
    print("✅ Dashboard generated")
    print("✅ Insights analysis completed")
    print("✅ All output files saved")
    print("\n🎉 CODEALPHA TASK 3 - DATA VISUALIZATION PROJECT COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()