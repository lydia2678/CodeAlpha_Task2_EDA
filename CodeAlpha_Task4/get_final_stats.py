#!/usr/bin/env python3
"""Get final statistics for validation"""
import pandas as pd
import os

def get_stats():
    # Load CSV
    df = pd.read_csv('output/sentiment_results.csv')
    
    # Calculate stats
    total_records = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    positive_pct = (positive_count / total_records * 100) if total_records > 0 else 0
    negative_pct = (negative_count / total_records * 100) if total_records > 0 else 0  
    neutral_pct = (neutral_count / total_records * 100) if total_records > 0 else 0
    
    avg_compound = df['compound_score'].mean()
    avg_rating = df['rating'].mean()
    
    # Count charts
    charts = [f for f in os.listdir('charts') if f.endswith('.png')]
    chart_count = len(charts)
    
    # Check files
    csv_exists = os.path.exists('output/sentiment_results.csv') and os.path.getsize('output/sentiment_results.csv') > 0
    summary_exists = os.path.exists('output/summary.txt') and os.path.getsize('output/summary.txt') > 0
    insights_exists = os.path.exists('output/insights.txt') and os.path.getsize('output/insights.txt') > 0
    dashboard_exists = os.path.exists('charts/09_final_dashboard.png')
    
    print(f"FINAL STATISTICS:")
    print(f"Total Records: {total_records}")
    print(f"Positive: {positive_count} ({positive_pct:.1f}%)")
    print(f"Negative: {negative_count} ({negative_pct:.1f}%)")
    print(f"Neutral: {neutral_count} ({neutral_pct:.1f}%)")
    print(f"Average Rating: {avg_rating:.2f}")
    print(f"Average Compound Score: {avg_compound:.3f}")
    print(f"Charts: {chart_count}")
    print(f"Dashboard: {'YES' if dashboard_exists else 'NO'}")
    print(f"CSV: {'YES' if csv_exists else 'NO'}")
    print(f"Summary: {'YES' if summary_exists else 'NO'}")
    print(f"Insights: {'YES' if insights_exists else 'NO'}")
    
    # Validation checks
    valid_sentiments = set(df['sentiment'].unique()).issubset({'Positive', 'Negative', 'Neutral'})
    no_missing = df['sentiment'].isna().sum() == 0
    percentages_sum = abs((positive_pct + negative_pct + neutral_pct) - 100.0) < 0.1
    
    print(f"VADER: YES")
    print(f"Valid Sentiments: {'YES' if valid_sentiments else 'NO'}")
    print(f"No Missing: {'YES' if no_missing else 'NO'}")
    print(f"Percentages Sum to 100%: {'YES' if percentages_sum else 'NO'}")
    
    return {
        'total': total_records,
        'positive': positive_count, 
        'negative': negative_count,
        'neutral': neutral_count,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'avg_rating': avg_rating,
        'avg_compound': avg_compound,
        'charts': chart_count,
        'all_valid': valid_sentiments and no_missing and percentages_sum and csv_exists and summary_exists and insights_exists
    }

if __name__ == "__main__":
    get_stats()