#!/usr/bin/env python3
"""Final validation of VADER sentiment analysis results"""
import pandas as pd
import os

def validate_vader_results():
    print("🔍 FINAL VALIDATION OF VADER SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Check CSV results
    try:
        df = pd.read_csv('output/sentiment_results.csv')
        print(f"📊 Results CSV: {len(df)} records loaded")
        
        # Check required columns
        required_cols = ['positive_score', 'negative_score', 'neutral_score', 'compound_score', 'sentiment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        print("✅ All required VADER columns present")
        
        # Validate VADER scores
        print(f"\n📈 VADER SCORE VALIDATION:")
        print(f"   Compound range: {df['compound_score'].min():.3f} to {df['compound_score'].max():.3f}")
        print(f"   Positive range: {df['positive_score'].min():.3f} to {df['positive_score'].max():.3f}")
        print(f"   Negative range: {df['negative_score'].min():.3f} to {df['negative_score'].max():.3f}")
        print(f"   Neutral range: {df['neutral_score'].min():.3f} to {df['neutral_score'].max():.3f}")
        
        # Validate sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        print(f"\n🎭 SENTIMENT DISTRIBUTION:")
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count/total*100) if total > 0 else 0
            print(f"   {sentiment}: {count} ({pct:.1f}%)")
        
        # Validate thresholds
        positive_check = (df[df['compound_score'] >= 0.05]['sentiment'] == 'Positive').all()
        negative_check = (df[df['compound_score'] <= -0.05]['sentiment'] == 'Negative').all()
        neutral_check = (df[(df['compound_score'] > -0.05) & (df['compound_score'] < 0.05)]['sentiment'] == 'Neutral').all()
        
        print(f"\n✅ CLASSIFICATION VALIDATION:")
        print(f"   Positive threshold (≥0.05): {'✓' if positive_check else '✗'}")
        print(f"   Negative threshold (≤-0.05): {'✓' if negative_check else '✗'}")
        print(f"   Neutral threshold (-0.05 to 0.05): {'✓' if neutral_check else '✗'}")
        
        # Check ratings
        if 'rating' in df.columns:
            avg_rating = df['rating'].mean()
            print(f"\n⭐ RATING ANALYSIS:")
            print(f"   Average Rating: {avg_rating:.2f}/5.0")
        
        # Check categories
        if 'product_category' in df.columns:
            categories = df['product_category'].value_counts()
            print(f"\n🏷️ CATEGORY DISTRIBUTION:")
            for cat, count in categories.items():
                print(f"   {cat}: {count} reviews")
        
        print(f"\n📊 KEY METRICS:")
        print(f"   Average Compound Score: {df['compound_score'].mean():.3f}")
        print(f"   Most Positive Review: {df['compound_score'].max():.3f}")
        print(f"   Most Negative Review: {df['compound_score'].min():.3f}")
        
    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        return False
    
    # Check charts
    required_charts = [
        '01_sentiment_distribution.png',
        '02_sentiment_percentage.png', 
        '03_rating_distribution.png',
        '04_sentiment_vs_rating.png',
        '05_sentiment_by_category.png',
        '06_sentiment_trend.png',
        '07_compound_score_distribution.png',
        '08_average_scores.png',
        '09_final_dashboard.png'
    ]
    
    print(f"\n📊 CHART VALIDATION:")
    chart_count = 0
    for chart in required_charts:
        if os.path.exists(f'charts/{chart}'):
            print(f"   ✓ {chart}")
            chart_count += 1
        else:
            print(f"   ✗ {chart} MISSING")
    
    print(f"\n📈 Charts Status: {chart_count}/9 required charts present")
    
    # Check output files
    output_files = ['sentiment_results.csv', 'summary.txt', 'insights.txt']
    print(f"\n💾 OUTPUT FILES:")
    for file in output_files:
        path = f'output/{file}'
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✓ {file} ({size} bytes)")
        else:
            print(f"   ✗ {file} MISSING")
    
    # Final assessment
    all_charts = chart_count >= 8
    has_vader_scores = all(col in df.columns for col in required_cols)
    valid_sentiments = set(df['sentiment'].unique()).issubset({'Positive', 'Negative', 'Neutral'})
    no_missing_sentiments = df['sentiment'].isna().sum() == 0
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   VADER Implementation: {'✅ PASS' if has_vader_scores else '❌ FAIL'}")
    print(f"   Sufficient Charts (≥8): {'✅ PASS' if all_charts else '❌ FAIL'}")
    print(f"   Valid Sentiments Only: {'✅ PASS' if valid_sentiments else '❌ FAIL'}")
    print(f"   No Missing Labels: {'✅ PASS' if no_missing_sentiments else '❌ FAIL'}")
    print(f"   Output Files: {'✅ PASS' if all(os.path.exists(f'output/{f}') for f in output_files) else '❌ FAIL'}")
    
    overall_pass = (has_vader_scores and all_charts and valid_sentiments and 
                   no_missing_sentiments and all(os.path.exists(f'output/{f}') for f in output_files))
    
    print(f"\n{'🎉 OVERALL STATUS: COMPLETE AND VALIDATED' if overall_pass else '⚠️  OVERALL STATUS: INCOMPLETE'}")
    
    return overall_pass

if __name__ == "__main__":
    validate_vader_results()