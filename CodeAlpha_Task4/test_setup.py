"""
Test script to verify basic setup and dependencies
"""
import sys
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print("✓ pandas available")
except ImportError:
    print("❌ pandas not available - run: pip install pandas")

try:
    import numpy as np
    print("✓ numpy available")
except ImportError:
    print("❌ numpy not available - run: pip install numpy")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError:
    print("❌ matplotlib not available - run: pip install matplotlib")

try:
    import seaborn as sns
    print("✓ seaborn available")
except ImportError:
    print("❌ seaborn not available - run: pip install seaborn")

try:
    import nltk
    print("✓ NLTK available")
except ImportError:
    print("❌ NLTK not available - run: pip install nltk")

# Test data loading
try:
    df = pd.read_csv('data/synthetic_reviews.csv')
    print(f"✓ Dataset loaded: {len(df)} records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Categories: {df['product_category'].unique()}")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")

print("\nSetup verification complete!")