#!/usr/bin/env python3
"""Minimal sentiment analysis to test core functionality"""
import pandas as pd
import os

def main():
    print("🔍 Running minimal sentiment analysis test...")
    
    try:
        # Create directories
        os.makedirs('charts', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/synthetic_reviews.csv')
        print(f"✓ Loaded {len(df)} reviews")
        
        # Basic sentiment analysis without NLTK for testing
        print("Performing basic sentiment analysis...")
        
        # Simple keyword-based sentiment (for testing)
        positive_words = ['amazing', 'excellent', 'great', 'love', 'perfect', 'outstanding', 'fantastic', 'incredible']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 'poor', 'bad']
        
        def simple_sentiment(text):
            if pd.isna(text):
                return 'Neutral', 0.0
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'Positive', 0.5
            elif neg_count > pos_count:
                return 'Negative', -0.5
            else:
                return 'Neutral', 0.0
        
        # Apply sentiment analysis
        sentiments = []
        compound_scores = []
        
        for text in df['review_text']:
            sentiment, score = simple_sentiment(text)
            sentiments.append(sentiment)
            compound_scores.append(score)
        
        df['sentiment'] = sentiments
        df['compound_score'] = compound_scores
        df['cleaned_text'] = df['review_text'].str.strip()
        
        # Generate basic statistics
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        print(f"\n📊 Results:")
        print(f"Total reviews: {total}")
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count/total*100) if total > 0 else 0
            print(f"{sentiment}: {count} ({pct:.1f}%)")
        
        # Save basic results
        output_df = df[['review_id', 'product_category', 'rating', 'review_text', 'cleaned_text', 'sentiment', 'compound_score']].copy()
        output_df.to_csv('output/sentiment_results.csv', index=False)
        print("✓ Saved sentiment_results.csv")
        
        # Create basic insights
        insights = f"""BASIC SENTIMENT ANALYSIS RESULTS
Total Reviews: {total}
Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total*100:.1f}%)
Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total*100:.1f}%)
Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total*100:.1f}%)

Note: This is a simplified analysis for testing purposes.
"""
        
        with open('output/insights.txt', 'w') as f:
            f.write(insights)
        print("✓ Saved insights.txt")
        
        # Create basic summary
        summary = f"""TECHNICAL SUMMARY
Dataset: {len(df)} product reviews
Method: Simple keyword-based sentiment analysis
Results: Successfully classified all reviews
Output: CSV file with sentiment classifications
"""
        
        with open('output/summary.txt', 'w') as f:
            f.write(summary)
        print("✓ Saved summary.txt")
        
        print("\n✅ Minimal analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()