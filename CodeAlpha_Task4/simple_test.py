#!/usr/bin/env python3
"""Simple test to verify basic functionality"""
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

try:
    import pandas as pd
    print("✓ pandas available")
except ImportError as e:
    print("❌ pandas not available:", str(e))
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy available")
except ImportError as e:
    print("❌ numpy not available:", str(e))
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError as e:
    print("❌ matplotlib not available:", str(e))
    sys.exit(1)

try:
    import seaborn as sns
    print("✓ seaborn available")
except ImportError as e:
    print("❌ seaborn not available:", str(e))
    sys.exit(1)

try:
    import nltk
    print("✓ NLTK available")
    from nltk.sentiment import SentimentIntensityAnalyzer
    print("✓ VADER sentiment analyzer available")
except ImportError as e:
    print("❌ NLTK/VADER not available:", str(e))
    sys.exit(1)

# Test data loading
try:
    if os.path.exists('data/synthetic_reviews.csv'):
        df = pd.read_csv('data/synthetic_reviews.csv')
        print(f"✓ Dataset loaded successfully: {len(df)} records")
    else:
        print("❌ Dataset file not found")
        sys.exit(1)
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    sys.exit(1)

print("✅ All basic components are working!")