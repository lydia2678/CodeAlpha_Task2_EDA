#!/usr/bin/env python3
"""Create basic visualizations for sentiment analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_charts():
    print("📊 Creating visualization charts...")
    
    try:
        # Load results
        df = pd.read_csv('output/sentiment_results.csv')
        
        # Ensure charts directory
        os.makedirs('charts', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Sentiment Distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2E8B57', '#DC143C', '#4682B4']  # Green, Red, Blue
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title('Sentiment Distribution - Review Counts', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('charts/01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 01_sentiment_distribution.png")
        
        # 2. Sentiment Percentage
        plt.figure(figsize=(10, 8))
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12})
        plt.title('Sentiment Distribution - Percentages', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('charts/02_sentiment_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 02_sentiment_percentage.png")
        
        # 3. Rating Distribution
        if 'rating' in df.columns:
            plt.figure(figsize=(10, 6))
            rating_counts = df['rating'].value_counts().sort_index()
            bars = plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
            plt.title('Rating Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Rating', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig('charts/03_rating_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created 03_rating_distribution.png")
        
        # 4. Category Distribution
        if 'product_category' in df.columns:
            plt.figure(figsize=(12, 6))
            category_counts = df['product_category'].value_counts()
            bars = plt.bar(category_counts.index, category_counts.values, color='orange')
            plt.title('Reviews by Product Category', fontsize=16, fontweight='bold')
            plt.xlabel('Category', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('charts/04_category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created 04_category_distribution.png")
        
        # 5. Simple Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sentiment distribution
        sentiment_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Sentiment pie
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Sentiment Percentages', fontweight='bold')
        
        # Rating distribution
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            rating_counts.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Rating Distribution', fontweight='bold')
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=0)
        
        # Category distribution
        if 'product_category' in df.columns:
            category_counts = df['product_category'].value_counts()
            category_counts.plot(kind='bar', ax=ax4, color='orange')
            ax4.set_title('Category Distribution', fontweight='bold')
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Sentiment Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('charts/05_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 05_dashboard.png")
        
        print(f"\n🎨 Successfully created 5 visualization charts!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_charts()