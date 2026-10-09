#!/usr/bin/env python3
"""Run the complete sentiment analysis with dependency installation"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'nltk']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} already available")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Download NLTK data
    try:
        import nltk
        print("Downloading NLTK resources...")
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK resources downloaded")
    except Exception as e:
        print(f"Warning: NLTK download issue: {e}")

def main():
    print("🚀 Starting CodeAlpha Task 4 Sentiment Analysis")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Import and run the analysis
    try:
        from task4_sentiment_analysis import SentimentAnalyzer
        
        print("\n📊 Running sentiment analysis...")
        analyzer = SentimentAnalyzer()
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\n✅ Analysis completed successfully!")
            return True
        else:
            print("\n❌ Analysis failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Task 4 completed successfully!")
    else:
        print("\n⚠️ Task 4 failed - check errors above")