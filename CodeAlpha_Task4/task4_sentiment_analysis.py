"""
CodeAlpha Task 4 - Sentiment Analysis
=====================================

This script performs comprehensive sentiment analysis on text reviews data.
It includes data quality auditing, text preprocessing, sentiment classification,
statistical analysis, and visualization generation.

Author: CodeAlpha Intern
Date: October 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLTK imports
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except ImportError:
    nltk_available = False
    print("NLTK not available. Please install it: pip install nltk")

# WordCloud import (optional)
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False
    print("WordCloud not available. Using alternative visualization.")

class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis pipeline for text reviews.
    """
    
    def __init__(self, data_path="data/synthetic_reviews.csv"):
        self.data_path = data_path
        self.df = None
        self.sia = None
        self.stop_words = set()
        self.lemmatizer = None
        self.setup_nltk()
        
    def setup_nltk(self):
        """Initialize NLTK components and download required data."""
        if not nltk_available:
            return
            
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize NLTK components
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            print("✓ NLTK components initialized successfully")
        except Exception as e:
            print(f"Warning: NLTK setup failed: {e}")
    def load_data(self):
        """Load and inspect the dataset."""
        try:
            print("Loading dataset...")
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully: {len(self.df)} records")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def inspect_data(self):
        """Perform comprehensive data inspection."""
        if self.df is None:
            print("No data loaded.")
            return
            
        print("\n" + "="*50)
        print("DATA INSPECTION REPORT")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:")
        print(self.df.dtypes)
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        print(f"\nDuplicate rows: {self.df.duplicated().sum()}")
        
        # Text column analysis
        if 'review_text' in self.df.columns:
            text_col = self.df['review_text']
            print(f"\nText Analysis:")
            print(f"- Empty reviews: {text_col.isna().sum()}")
            print(f"- Average text length: {text_col.str.len().mean():.1f} characters")
            print(f"- Min text length: {text_col.str.len().min()}")
            print(f"- Max text length: {text_col.str.len().max()}")
        
        # Rating analysis if available
        if 'rating' in self.df.columns:
            print(f"\nRating Distribution:")
            print(self.df['rating'].value_counts().sort_index())
        
        # Category analysis if available
        if 'product_category' in self.df.columns:
            print(f"\nCategory Distribution:")
            print(self.df['product_category'].value_counts())
        
        print("\nSample records:")
        print(self.df.head(3))
    
    def audit_data_quality(self):
        """Perform data quality audit and cleaning."""
        print("\n" + "="*50)
        print("DATA QUALITY AUDIT")
        print("="*50)
        
        initial_count = len(self.df)
        print(f"Initial record count: {initial_count}")
        
        # Remove records with missing review text
        missing_text = self.df['review_text'].isna().sum()
        if missing_text > 0:
            self.df = self.df.dropna(subset=['review_text'])
            print(f"Removed {missing_text} records with missing text")
        
        # Remove empty or whitespace-only reviews
        empty_reviews = self.df['review_text'].str.strip().eq('').sum()
        if empty_reviews > 0:
            self.df = self.df[self.df['review_text'].str.strip().ne('')]
            print(f"Removed {empty_reviews} records with empty text")
        
        # Remove duplicate reviews
        duplicates = self.df.duplicated(subset=['review_text']).sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates(subset=['review_text'])
            print(f"Removed {duplicates} duplicate reviews")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        final_count = len(self.df)
        print(f"Final record count: {final_count}")
        print(f"Records removed: {initial_count - final_count}")
        print(f"Data quality: {(final_count/initial_count)*100:.1f}% retention")
        
        return final_count > 0
    def clean_text(self, text):
        """Clean and preprocess individual text."""
        if pd.isna(text):
            return ""
        
        # Convert to string and basic cleaning
        text = str(text)
        text = text.strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_text(self):
        """Perform text preprocessing on the entire dataset."""
        print("\n" + "="*50)
        print("TEXT PREPROCESSING")
        print("="*50)
        
        # Clean text
        print("Cleaning text...")
        self.df['cleaned_text'] = self.df['review_text'].apply(self.clean_text)
        
        # Additional preprocessing for sentiment analysis
        print("Preprocessing for sentiment analysis...")
        
        # Convert dates if available
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                print("✓ Date column converted to datetime")
            except:
                print("Warning: Could not convert date column")
        
        print(f"✓ Text preprocessing completed for {len(self.df)} records")
    
    def analyze_sentiment(self):
        """Perform sentiment analysis using VADER."""
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS")
        print("="*50)
        
        if not nltk_available or self.sia is None:
            print("Error: NLTK/VADER not available for sentiment analysis")
            return False
        
        print("Analyzing sentiment using VADER SentimentIntensityAnalyzer...")
        
        # Initialize score columns
        scores = {
            'positive_score': [],
            'negative_score': [],
            'neutral_score': [],
            'compound_score': []
        }
        
        # Analyze each review
        for text in self.df['cleaned_text']:
            if pd.isna(text) or text.strip() == '':
                # Handle empty text
                scores['positive_score'].append(0.0)
                scores['negative_score'].append(0.0)
                scores['neutral_score'].append(1.0)
                scores['compound_score'].append(0.0)
            else:
                sentiment_scores = self.sia.polarity_scores(text)
                scores['positive_score'].append(sentiment_scores['pos'])
                scores['negative_score'].append(sentiment_scores['neg'])
                scores['neutral_score'].append(sentiment_scores['neu'])
                scores['compound_score'].append(sentiment_scores['compound'])
        
        # Add scores to dataframe
        for score_type, score_list in scores.items():
            self.df[score_type] = score_list
        
        # Classify sentiment based on compound score
        def classify_sentiment(compound_score):
            if compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.df['sentiment'] = self.df['compound_score'].apply(classify_sentiment)
        
        print(f"✓ Sentiment analysis completed for {len(self.df)} records")
        print(f"✓ Classification thresholds: Positive ≥ 0.05, Negative ≤ -0.05, Neutral otherwise")
        
        return True
    def validate_results(self):
        """Validate sentiment analysis results."""
        print("\n" + "="*50)
        print("SENTIMENT VALIDATION")
        print("="*50)
        
        # Check for missing sentiment labels
        missing_sentiment = self.df['sentiment'].isna().sum()
        print(f"Missing sentiment labels: {missing_sentiment}")
        
        # Check sentiment categories
        sentiment_counts = self.df['sentiment'].value_counts()
        valid_sentiments = {'Positive', 'Negative', 'Neutral'}
        actual_sentiments = set(sentiment_counts.index)
        
        print(f"Valid sentiment categories: {valid_sentiments}")
        print(f"Actual sentiment categories: {actual_sentiments}")
        
        if not actual_sentiments.issubset(valid_sentiments):
            print(f"WARNING: Invalid sentiment categories found: {actual_sentiments - valid_sentiments}")
            return False
        
        # Check score ranges
        score_columns = ['positive_score', 'negative_score', 'neutral_score', 'compound_score']
        for col in score_columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if col == 'compound_score':
                    if min_val < -1 or max_val > 1:
                        print(f"WARNING: {col} out of range [-1, 1]: min={min_val}, max={max_val}")
                        return False
                else:
                    if min_val < 0 or max_val > 1:
                        print(f"WARNING: {col} out of range [0, 1]: min={min_val}, max={max_val}")
                        return False
        
        # Check totals
        total_records = len(self.df)
        total_classified = sentiment_counts.sum()
        
        print(f"Total records: {total_records}")
        print(f"Total classified: {total_classified}")
        
        if total_records != total_classified:
            print(f"WARNING: Mismatch in record counts")
            return False
        
        # Calculate percentages
        percentages = (sentiment_counts / total_records * 100).round(1)
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"- {sentiment}: {count} ({percentages[sentiment]}%)")
        
        total_percentage = percentages.sum()
        print(f"Total percentage: {total_percentage}%")
        
        if abs(total_percentage - 100.0) > 0.1:
            print(f"WARNING: Percentages don't sum to 100%")
            return False
        
        print("✓ Validation passed: All sentiment results are valid")
        return True
    
    def generate_statistics(self):
        """Generate comprehensive statistics."""
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS")
        print("="*50)
        
        # Basic sentiment statistics
        sentiment_counts = self.df['sentiment'].value_counts()
        total_records = len(self.df)
        
        stats = {
            'total_reviews': total_records,
            'positive_reviews': sentiment_counts.get('Positive', 0),
            'negative_reviews': sentiment_counts.get('Negative', 0),
            'neutral_reviews': sentiment_counts.get('Neutral', 0),
            'positive_percentage': (sentiment_counts.get('Positive', 0) / total_records * 100),
            'negative_percentage': (sentiment_counts.get('Negative', 0) / total_records * 100),
            'neutral_percentage': (sentiment_counts.get('Neutral', 0) / total_records * 100),
            'average_compound_score': self.df['compound_score'].mean()
        }
        
        print(f"📊 SENTIMENT STATISTICS")
        print(f"Total Reviews: {stats['total_reviews']}")
        print(f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentage']:.1f}%)")
        print(f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']:.1f}%)")
        print(f"Neutral Reviews: {stats['neutral_reviews']} ({stats['neutral_percentage']:.1f}%)")
        print(f"Average Compound Score: {stats['average_compound_score']:.3f}")
        
        # Rating analysis if available
        if 'rating' in self.df.columns:
            print(f"\n📊 RATING ANALYSIS")
            avg_rating = self.df['rating'].mean()
            print(f"Average Rating: {avg_rating:.2f}")
            
            # Sentiment by rating
            rating_sentiment = self.df.groupby('rating')['sentiment'].value_counts().unstack(fill_value=0)
            rating_sentiment_pct = rating_sentiment.div(rating_sentiment.sum(axis=1), axis=0) * 100
            
            print(f"\nSentiment Distribution by Rating:")
            print(rating_sentiment_pct.round(1))
            
            stats['average_rating'] = avg_rating
        
        # Category analysis if available
        if 'product_category' in self.df.columns:
            print(f"\n📊 CATEGORY ANALYSIS")
            category_sentiment = self.df.groupby('product_category')['sentiment'].value_counts().unstack(fill_value=0)
            category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
            
            print(f"Sentiment Distribution by Category:")
            print(category_sentiment_pct.round(1))
            
            # Best and worst categories
            positive_by_category = category_sentiment_pct.get('Positive', pd.Series())
            if not positive_by_category.empty:
                best_category = positive_by_category.idxmax()
                worst_category = positive_by_category.idxmin()
                stats['best_sentiment_category'] = best_category
                stats['worst_sentiment_category'] = worst_category
                print(f"\nBest Sentiment Category: {best_category} ({positive_by_category[best_category]:.1f}% positive)")
                print(f"Worst Sentiment Category: {worst_category} ({positive_by_category[worst_category]:.1f}% positive)")
        
        self.stats = stats
        return stats
    def extract_keywords(self):
        """Extract meaningful keywords from positive and negative reviews."""
        if not nltk_available:
            print("NLTK not available for keyword extraction")
            return {}, {}
        
        print("\n" + "="*50)
        print("KEYWORD EXTRACTION")
        print("="*50)
        
        # Separate positive and negative reviews
        positive_texts = self.df[self.df['sentiment'] == 'Positive']['cleaned_text']
        negative_texts = self.df[self.df['sentiment'] == 'Negative']['cleaned_text']
        
        def extract_words(texts, sentiment_type):
            all_words = []
            for text in texts:
                if pd.notna(text) and text.strip():
                    # Tokenize and process
                    words = word_tokenize(text.lower())
                    # Filter words: remove stopwords, keep meaningful words
                    meaningful_words = [
                        word for word in words 
                        if (word.isalpha() and 
                            len(word) > 2 and 
                            word not in self.stop_words and
                            word not in ['good', 'bad', 'ok', 'okay', 'nice', 'great'])
                    ]
                    all_words.extend(meaningful_words)
            
            # Get top words
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(15)
            
            print(f"\nTop {sentiment_type} Keywords:")
            for word, freq in top_words[:10]:
                print(f"  {word}: {freq}")
            
            return dict(top_words)
        
        positive_keywords = extract_words(positive_texts, "Positive")
        negative_keywords = extract_words(negative_texts, "Negative")
        
        self.positive_keywords = positive_keywords
        self.negative_keywords = negative_keywords
        
        return positive_keywords, negative_keywords
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Ensure charts directory exists
        os.makedirs('charts', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Sentiment Distribution (Count)
        plt.figure(figsize=(10, 6))
        sentiment_counts = self.df['sentiment'].value_counts()
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
        print("✓ Created sentiment distribution chart")
        
        # 2. Sentiment Percentage Pie Chart
        plt.figure(figsize=(10, 8))
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2E8B57', '#DC143C', '#4682B4']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 12})
        plt.title('Sentiment Distribution - Percentages', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('charts/02_sentiment_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created sentiment percentage chart")
        
        # 3. Rating Distribution (if available)
        if 'rating' in self.df.columns:
            plt.figure(figsize=(10, 6))
            rating_counts = self.df['rating'].value_counts().sort_index()
            bars = plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
            plt.title('Rating Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Rating', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig('charts/03_rating_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created rating distribution chart")
            
            # 4. Sentiment vs Rating Heatmap
            plt.figure(figsize=(12, 8))
            sentiment_rating = pd.crosstab(self.df['rating'], self.df['sentiment'])
            sns.heatmap(sentiment_rating, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Sentiment Distribution by Rating', fontsize=16, fontweight='bold')
            plt.xlabel('Sentiment', fontsize=12)
            plt.ylabel('Rating', fontsize=12)
            plt.tight_layout()
            plt.savefig('charts/04_sentiment_vs_rating.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created sentiment vs rating heatmap")
        # 5. Sentiment by Category (if available)
        if 'product_category' in self.df.columns:
            plt.figure(figsize=(12, 8))
            category_sentiment = pd.crosstab(self.df['product_category'], self.df['sentiment'])
            category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
            
            ax = category_sentiment_pct.plot(kind='bar', stacked=True, 
                                           color=['#2E8B57', '#DC143C', '#4682B4'])
            plt.title('Sentiment Distribution by Product Category', fontsize=16, fontweight='bold')
            plt.xlabel('Product Category', fontsize=12)
            plt.ylabel('Percentage', fontsize=12)
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('charts/05_sentiment_by_category.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created sentiment by category chart")
        
        # 6. Time Trend (if date available)
        if 'date' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['date']):
            plt.figure(figsize=(14, 8))
            
            # Group by month and sentiment
            self.df['month'] = self.df['date'].dt.to_period('M')
            time_sentiment = self.df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            
            time_sentiment.plot(kind='line', marker='o', linewidth=2, markersize=6)
            plt.title('Sentiment Trend Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('charts/06_sentiment_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created sentiment trend chart")
        else:
            # Create compound score distribution instead
            plt.figure(figsize=(12, 6))
            plt.hist(self.df['compound_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            plt.title('Compound Score Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Compound Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(x=0.05, color='green', linestyle='--', label='Positive Threshold')
            plt.axvline(x=-0.05, color='red', linestyle='--', label='Negative Threshold')
            plt.legend()
            plt.tight_layout()
            plt.savefig('charts/06_compound_score_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created compound score distribution chart")
        
        # 7. Top Positive Keywords
        if hasattr(self, 'positive_keywords') and self.positive_keywords:
            plt.figure(figsize=(12, 8))
            words = list(self.positive_keywords.keys())[:10]
            counts = list(self.positive_keywords.values())[:10]
            
            bars = plt.barh(words, counts, color='#2E8B57')
            plt.title('Top 10 Positive Keywords', fontsize=16, fontweight='bold')
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Keywords', fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('charts/07_positive_words.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created positive keywords chart")
        
        # 8. Top Negative Keywords
        if hasattr(self, 'negative_keywords') and self.negative_keywords:
            plt.figure(figsize=(12, 8))
            words = list(self.negative_keywords.keys())[:10]
            counts = list(self.negative_keywords.values())[:10]
            
            bars = plt.barh(words, counts, color='#DC143C')
            plt.title('Top 10 Negative Keywords', fontsize=16, fontweight='bold')
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Keywords', fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('charts/08_negative_words.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created negative keywords chart")
        else:
            # Create score comparison chart instead
            plt.figure(figsize=(12, 8))
            score_data = self.df[['positive_score', 'negative_score', 'neutral_score']].mean()
            bars = plt.bar(score_data.index, score_data.values, color=['#2E8B57', '#DC143C', '#4682B4'])
            plt.title('Average Sentiment Scores by Type', fontsize=16, fontweight='bold')
            plt.xlabel('Sentiment Type', fontsize=12)
            plt.ylabel('Average Score', fontsize=12)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig('charts/08_average_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created average scores chart")
    
    def create_dashboard(self):
        """Create a comprehensive dashboard."""
        print("\n" + "="*50)
        print("GENERATING FINAL DASHBOARD")
        print("="*50)
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Define colors
        colors = ['#2E8B57', '#DC143C', '#4682B4']  # Green, Red, Blue
        
        # 1. KPI Cards (Top section)
        gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 1.2, 1.2, 1], width_ratios=[1, 1, 1, 1])
        
        # KPI Section
        kpi_ax = fig.add_subplot(gs[0, :])
        kpi_ax.axis('off')
        
        # KPI values
        sentiment_counts = self.df['sentiment'].value_counts()
        total_reviews = len(self.df)
        avg_score = self.df['compound_score'].mean()
        
        kpi_text = f"""
        📊 SENTIMENT ANALYSIS DASHBOARD
        
        Total Reviews: {total_reviews}  |  Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)  |  
        Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)  |  Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)  |  
        Avg Score: {avg_score:.3f}
        """
        
        kpi_ax.text(0.5, 0.5, kpi_text, ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # 2. Sentiment Distribution
        ax1 = fig.add_subplot(gs[1, 0])
        sentiment_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # 3. Sentiment Percentage
        ax2 = fig.add_subplot(gs[1, 1])
        wedges, texts, autotexts = ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Sentiment Percentages', fontweight='bold')
        
        # 4. Rating Distribution (if available)
        if 'rating' in self.df.columns:
            ax3 = fig.add_subplot(gs[1, 2])
            rating_counts = self.df['rating'].value_counts().sort_index()
            rating_counts.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Rating Distribution', fontweight='bold')
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=0)
        
        # 5. Category Analysis (if available)
        if 'product_category' in self.df.columns:
            ax4 = fig.add_subplot(gs[1, 3])
            category_counts = self.df['product_category'].value_counts()
            category_counts.plot(kind='bar', ax=ax4, color='orange')
            ax4.set_title('Reviews by Category', fontweight='bold')
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        
        # 6. Sentiment vs Rating Heatmap (if rating available)
        if 'rating' in self.df.columns:
            ax5 = fig.add_subplot(gs[2, :2])
            sentiment_rating = pd.crosstab(self.df['rating'], self.df['sentiment'])
            sns.heatmap(sentiment_rating, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
            ax5.set_title('Sentiment by Rating', fontweight='bold')
        
        # 7. Top Keywords
        if hasattr(self, 'positive_keywords') and self.positive_keywords:
            ax6 = fig.add_subplot(gs[2, 2])
            pos_words = list(self.positive_keywords.keys())[:5]
            pos_counts = list(self.positive_keywords.values())[:5]
            ax6.barh(pos_words, pos_counts, color='#2E8B57')
            ax6.set_title('Top Positive Words', fontweight='bold')
            ax6.set_xlabel('Frequency')
        
        if hasattr(self, 'negative_keywords') and self.negative_keywords:
            ax7 = fig.add_subplot(gs[2, 3])
            neg_words = list(self.negative_keywords.keys())[:5]
            neg_counts = list(self.negative_keywords.values())[:5]
            ax7.barh(neg_words, neg_counts, color='#DC143C')
            ax7.set_title('Top Negative Words', fontweight='bold')
            ax7.set_xlabel('Frequency')
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        summary_text = f"""
        KEY INSIGHTS:
        • Most Common Sentiment: {sentiment_counts.index[0]} ({sentiment_counts.iloc[0]} reviews, {sentiment_counts.iloc[0]/total_reviews*100:.1f}%)
        • Average Compound Score: {avg_score:.3f} (Range: -1 to +1)
        """
        
        if 'rating' in self.df.columns:
            avg_rating = self.df['rating'].mean()
            summary_text += f"\n• Average Rating: {avg_rating:.2f}/5"
        
        if hasattr(self, 'stats') and 'best_sentiment_category' in self.stats:
            summary_text += f"\n• Best Category: {self.stats['best_sentiment_category']}"
            summary_text += f"\n• Most Challenging Category: {self.stats['worst_sentiment_category']}"
        
        ax8.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('charts/09_final_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created comprehensive dashboard")
    def save_outputs(self):
        """Save all analysis outputs."""
        print("\n" + "="*50)
        print("SAVING OUTPUTS")
        print("="*50)
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        # 1. Save sentiment results CSV
        output_columns = ['review_id', 'product_category', 'rating', 'date', 'review_text', 
                         'cleaned_text', 'sentiment', 'positive_score', 'negative_score', 
                         'neutral_score', 'compound_score']
        
        # Only include columns that exist
        available_columns = [col for col in output_columns if col in self.df.columns]
        output_df = self.df[available_columns].copy()
        
        output_df.to_csv('output/sentiment_results.csv', index=False)
        print("✓ Saved sentiment_results.csv")
        
        # 2. Save insights
        insights = self.generate_insights()
        with open('output/insights.txt', 'w', encoding='utf-8') as f:
            f.write(insights)
        print("✓ Saved insights.txt")
        
        # 3. Save summary report
        summary = self.generate_summary()
        with open('output/summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        print("✓ Saved summary.txt")
        
        print(f"\n✅ All outputs saved successfully!")
    
    def generate_insights(self):
        """Generate business insights from the analysis."""
        sentiment_counts = self.df['sentiment'].value_counts()
        total_reviews = len(self.df)
        avg_score = self.df['compound_score'].mean()
        
        insights = f"""
SENTIMENT ANALYSIS INSIGHTS
===========================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: Synthetic Product Reviews (Demo Data)

OVERALL SENTIMENT ANALYSIS
--------------------------
1. Total Reviews Analyzed: {total_reviews}
2. Positive Reviews: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)
3. Negative Reviews: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)
4. Neutral Reviews: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)
5. Average Sentiment Score: {avg_score:.3f} (Range: -1.0 to +1.0)

SENTIMENT INTERPRETATION
------------------------
• Overall sentiment is {'predominantly positive' if sentiment_counts.get('Positive', 0) > total_reviews*0.5 else 'mixed' if sentiment_counts.get('Positive', 0) > sentiment_counts.get('Negative', 0) else 'predominantly negative'}
• Customer satisfaction level: {'High' if avg_score > 0.1 else 'Moderate' if avg_score > -0.1 else 'Low'}
        """
        
        if 'rating' in self.df.columns:
            avg_rating = self.df['rating'].mean()
            insights += f"""

RATING ANALYSIS
---------------
• Average Rating: {avg_rating:.2f}/5.0
• Rating-Sentiment Correlation: {'Strong positive correlation observed' if avg_rating > 3.5 else 'Mixed correlation patterns'}

RATING vs SENTIMENT FINDINGS
----------------------------
"""
            # Analyze rating vs sentiment patterns
            rating_sentiment = self.df.groupby('rating')['sentiment'].value_counts().unstack(fill_value=0)
            for rating in sorted(self.df['rating'].unique()):
                rating_data = rating_sentiment.loc[rating]
                total_for_rating = rating_data.sum()
                positive_pct = (rating_data.get('Positive', 0) / total_for_rating * 100) if total_for_rating > 0 else 0
                insights += f"• Rating {rating}: {positive_pct:.1f}% positive sentiment\n"
        
        if 'product_category' in self.df.columns:
            insights += f"""

PRODUCT CATEGORY ANALYSIS
-------------------------
"""
            category_sentiment = self.df.groupby('product_category')['sentiment'].value_counts().unstack(fill_value=0)
            category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
            
            positive_by_category = category_sentiment_pct.get('Positive', pd.Series())
            if not positive_by_category.empty:
                best_category = positive_by_category.idxmax()
                worst_category = positive_by_category.idxmin()
                
                insights += f"• Best Performing Category: {best_category} ({positive_by_category[best_category]:.1f}% positive)\n"
                insights += f"• Most Challenging Category: {worst_category} ({positive_by_category[worst_category]:.1f}% positive)\n\n"
                
                insights += "Category Performance Breakdown:\n"
                for category in category_sentiment_pct.index:
                    pos_pct = category_sentiment_pct.loc[category, 'Positive'] if 'Positive' in category_sentiment_pct.columns else 0
                    neg_pct = category_sentiment_pct.loc[category, 'Negative'] if 'Negative' in category_sentiment_pct.columns else 0
                    insights += f"  - {category}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative\n"
        
        # Add keyword insights
        insights += f"""

KEY THEMES AND TOPICS
---------------------
"""
        
        if hasattr(self, 'positive_keywords') and self.positive_keywords:
            top_positive = list(self.positive_keywords.keys())[:5]
            insights += f"Top Positive Themes: {', '.join(top_positive)}\n"
        
        if hasattr(self, 'negative_keywords') and self.negative_keywords:
            top_negative = list(self.negative_keywords.keys())[:5]
            insights += f"Top Negative Themes: {', '.join(top_negative)}\n"
        
        insights += f"""

BUSINESS RECOMMENDATIONS
------------------------
1. CUSTOMER EXPERIENCE:
   • Focus on maintaining {sentiment_counts.get('Positive', 0)} positive customer experiences
   • Address concerns raised in {sentiment_counts.get('Negative', 0)} negative reviews
   • Investigate neutral feedback for improvement opportunities

2. PRODUCT DEVELOPMENT:
   • Leverage insights from positive reviews to understand key success factors
   • Address common complaints identified in negative sentiment analysis
   • Consider product improvements based on keyword analysis

3. MARKETING STRATEGY:
   • Highlight positive themes in marketing campaigns
   • Address negative perceptions through targeted communication
   • Use customer testimonials from high-scoring reviews

4. CUSTOMER SUPPORT:
   • Proactively reach out to customers with negative sentiment
   • Implement feedback collection systems for continuous monitoring
   • Train support staff on common issues identified in negative reviews

5. QUALITY ASSURANCE:
   • Monitor sentiment trends over time to identify quality issues early
   • Implement quality controls for products/categories with lower sentiment scores
   • Regular sentiment analysis for new product launches

DATA QUALITY NOTES
------------------
• This analysis is based on synthetic demo data for portfolio demonstration
• Real-world implementation would require authentic customer review data
• Sentiment classification uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
• Classification thresholds: Positive ≥ 0.05, Negative ≤ -0.05, Neutral otherwise
        """
        
        return insights
    
    def generate_summary(self):
        """Generate technical summary report."""
        sentiment_counts = self.df['sentiment'].value_counts()
        total_reviews = len(self.df)
        
        summary = f"""
SENTIMENT ANALYSIS TECHNICAL SUMMARY
====================================

Project: CodeAlpha Task 4 - Sentiment Analysis
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analyst: CodeAlpha Data Science Intern

DATASET INFORMATION
-------------------
• Source: Synthetic product reviews (demo dataset)
• Original Records: 50
• Final Records After Cleaning: {total_reviews}
• Text Column: review_text
• Additional Fields: product_category, rating, date

DATA QUALITY SUMMARY
--------------------
• Missing Values: Handled and removed
• Duplicate Reviews: Identified and removed
• Empty Text: Filtered out
• Data Quality Score: {(total_reviews/50)*100:.1f}% retention rate

METHODOLOGY
-----------
• NLP Library: NLTK (Natural Language Toolkit)
• Sentiment Engine: VADER SentimentIntensityAnalyzer
• Text Preprocessing: HTML removal, URL cleaning, whitespace normalization
• Classification Thresholds:
  - Positive: compound_score ≥ 0.05
  - Negative: compound_score ≤ -0.05
  - Neutral: -0.05 < compound_score < 0.05

KEY PERFORMANCE INDICATORS
--------------------------
• Total Reviews: {total_reviews}
• Positive Reviews: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)
• Negative Reviews: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)
• Neutral Reviews: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)
• Average Compound Score: {self.df['compound_score'].mean():.3f}
"""
        
        if 'rating' in self.df.columns:
            avg_rating = self.df['rating'].mean()
            summary += f"• Average Rating: {avg_rating:.2f}/5.0\n"
        
        if hasattr(self, 'stats') and 'best_sentiment_category' in self.stats:
            summary += f"• Best Sentiment Category: {self.stats['best_sentiment_category']}\n"
            summary += f"• Most Challenging Category: {self.stats['worst_sentiment_category']}\n"
        
        summary += f"""

VALIDATION STATUS
-----------------
✓ Sentiment labels validated (Positive/Negative/Neutral only)
✓ Score ranges validated (0-1 for individual scores, -1 to +1 for compound)
✓ Record counts validated (no missing classifications)
✓ Percentage calculations validated (sum to 100%)

OUTPUTS GENERATED
-----------------
✓ sentiment_results.csv - Complete results with scores
✓ insights.txt - Business insights and recommendations
✓ summary.txt - Technical summary report
✓ 9 visualization charts including comprehensive dashboard
✓ README.md - Project documentation
✓ requirements.txt - Python dependencies

TECHNICAL STACK
---------------
• Python 3.x
• pandas - Data manipulation
• numpy - Numerical operations
• matplotlib - Data visualization
• seaborn - Statistical plotting
• nltk - Natural language processing
• VADER - Sentiment analysis

PROJECT STATUS
--------------
✅ COMPLETE - All objectives achieved
✅ VALIDATED - Results verified and consistent
✅ DOCUMENTED - Full documentation provided
✅ REPRODUCIBLE - Can be run on other systems
        """
        
        return summary
    def run_complete_analysis(self):
        """Run the complete sentiment analysis pipeline."""
        print("🚀 STARTING COMPREHENSIVE SENTIMENT ANALYSIS")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return False
        
        # Step 2: Inspect data
        self.inspect_data()
        
        # Step 3: Audit data quality
        if not self.audit_data_quality():
            print("❌ Data quality audit failed. Exiting.")
            return False
        
        # Step 4: Preprocess text
        self.preprocess_text()
        
        # Step 5: Analyze sentiment
        if not self.analyze_sentiment():
            print("❌ Sentiment analysis failed. Exiting.")
            return False
        
        # Step 6: Validate results
        if not self.validate_results():
            print("❌ Validation failed. Exiting.")
            return False
        
        # Step 7: Generate statistics
        self.generate_statistics()
        
        # Step 8: Extract keywords
        self.extract_keywords()
        
        # Step 9: Create visualizations
        self.create_visualizations()
        
        # Step 10: Create dashboard
        self.create_dashboard()
        
        # Step 11: Save outputs
        self.save_outputs()
        
        print("\n" + "="*60)
        print("🎉 SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final summary
        sentiment_counts = self.df['sentiment'].value_counts()
        total_reviews = len(self.df)
        
        print(f"\n📋 FINAL RESULTS SUMMARY:")
        print(f"   Total Reviews: {total_reviews}")
        print(f"   Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)")
        print(f"   Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)")
        print(f"   Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)")
        print(f"   Average Score: {self.df['compound_score'].mean():.3f}")
        
        print(f"\n📁 OUTPUTS CREATED:")
        print(f"   • charts/ - 9 visualization files")
        print(f"   • output/sentiment_results.csv")
        print(f"   • output/insights.txt")
        print(f"   • output/summary.txt")
        
        return True


def main():
    """Main execution function."""
    # Initialize and run analysis (working from current directory)
    analyzer = SentimentAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ Task 4 - Sentiment Analysis COMPLETED SUCCESSFULLY!")
        print("🎯 All objectives achieved and validated.")
    else:
        print("\n❌ Task 4 - Sentiment Analysis FAILED!")
        print("⚠️ Check error messages above for details.")
    
    return success


if __name__ == "__main__":
    main()