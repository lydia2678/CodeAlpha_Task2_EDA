#!/usr/bin/env python3
"""
Simple runner for CodeAlpha Task 3 - Data Visualization
This script attempts to import required libraries and run the visualization
"""

print("🚀 CodeAlpha Task 3 - Data Visualization")
print("=" * 50)

# Check and install required packages
required_packages = {
    'pandas': 'pd',
    'numpy': 'np', 
    'matplotlib.pyplot': 'plt',
    'seaborn': 'sns'
}

missing_packages = []

for package, alias in required_packages.items():
    try:
        if package == 'matplotlib.pyplot':
            import matplotlib.pyplot as plt
            print(f"✓ matplotlib imported")
        elif package == 'pandas':
            import pandas as pd
            print(f"✓ pandas imported")  
        elif package == 'numpy':
            import numpy as np
            print(f"✓ numpy imported")
        elif package == 'seaborn':
            import seaborn as sns
            print(f"✓ seaborn imported")
    except ImportError:
        missing_packages.append(package)
        print(f"❌ {package} not available")

if missing_packages:
    print(f"\nMissing packages: {missing_packages}")
    print("Please install them using: pip install pandas numpy matplotlib seaborn")
    exit(1)

# If all packages are available, run the main visualization script
print("\n✅ All required packages are available!")
print("Running main visualization script...")

try:
    exec(open('task3_visualization.py').read())
except FileNotFoundError:
    print("❌ task3_visualization.py not found!")
except Exception as e:
    print(f"❌ Error running visualization: {e}")
    import traceback
    traceback.print_exc()