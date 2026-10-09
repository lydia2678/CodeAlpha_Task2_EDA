#!/usr/bin/env python3
"""Validate the sentiment analysis results"""
import pandas as pd
import os

def validate_results():
    print("🔍 VALIDATING SENTIMENT ANALYSIS RESULTS")
    print("=" * 50)
    
    # Check if results file exists
    if not os.path.exists('output/sentiment_results.csv'):
        print("❌ sentiment_results.csv not found")
        return False
    
    # Load and validate results
    df = pd.read_csv('output/sentiment_results.csv')
    
    print(f"📊 Dataset Validation:")
    print(f"   Total Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check required columns
    required_columns = ['review_text', 'cleaned_text', 'sentiment', 'compound_score']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    print("✓ All required columns present")
    
    # Check sentiment values
    sentiments = df['sentiment'].value_counts().to_dict()
    print(f"\n📈 Sentiment Distribution:")
    total = len(df)
    for sentiment, count in sentiments.items():
        pct = (count/total*100) if total > 0 else 0
        print(f"   {sentiment}: {count} ({pct:.1f}%)")
    
    # Validate sentiment categories
    valid_sentiments = {'Positive', 'Negative', 'Neutral'}
    actual_sentiments = set(df['sentiment'].unique())
    if not actual_sentiments.issubset(valid_sentiments):
        invalid = actual_sentiments - valid_sentiments
        print(f"❌ Invalid sentiment categories found: {invalid}")
        return False
    print("✓ Valid sentiment categories only")
    
    # Check for missing sentiment labels
    missing_sentiments = df['sentiment'].isna().sum()
    if missing_sentiments > 0:
        print(f"❌ Missing sentiment labels: {missing_sentiments}")
        return False
    print("✓ No missing sentiment labels")
    
    # Check percentages sum to 100%
    total_percentage = sum(sentiments.values()) / total * 100
    if abs(total_percentage - 100.0) > 0.1:
        print(f"❌ Percentages don't sum to 100%: {total_percentage}")
        return False
    print("✓ Percentages sum to 100%")
    
    # Check rating data
    if 'rating' in df.columns:
        avg_rating = df['rating'].mean()
        print(f"\n⭐ Rating Analysis:")
        print(f"   Average Rating: {avg_rating:.2f}")
        print(f"   Rating Range: {df['rating'].min()} - {df['rating'].max()}")
    
    # Check categories
    if 'product_category' in df.columns:
        categories = df['product_category'].value_counts().to_dict()
        print(f"\n🏷️ Category Distribution:")
        for category, count in categories.items():
            print(f"   {category}: {count}")
    
    print(f"\n✅ VALIDATION PASSED")
    print(f"   All {len(df)} records have valid sentiment classifications")
    return True

def validate_output_files():
    print(f"\n📁 OUTPUT FILE VALIDATION:")
    
    required_files = [
        'output/sentiment_results.csv',
        'output/insights.txt', 
        'output/summary.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✓ {file_path} ({size} bytes)")
        else:
            print(f"   ❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    try:
        results_valid = validate_results()
        files_valid = validate_output_files()
        
        if results_valid and files_valid:
            print(f"\n🎉 ALL VALIDATIONS PASSED!")
            print(f"   Sentiment analysis completed successfully")
            return True
        else:
            print(f"\n⚠️ VALIDATION FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()