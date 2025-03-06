import json
from transformers import pipeline

# Load pre-trained sentiment and emotion models
print("🔄 Loading sentiment and emotion analysis models...")

try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()

def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)
        return result[0]['label'], result[0]['score']
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return None, None

def analyze_emotions(text):
    try:
        # This fetches scores for all emotions
        all_emotions = emotion_pipeline(text, return_all_scores=True)[0]

        # Sort and select top 3 emotions
        top_emotions = sorted(all_emotions, key=lambda x: -x['score'])[:3]

        # Return them in label-score format
        return [(emo['label'], emo['score']) for emo in top_emotions]
    except Exception as e:
        print(f"❌ Emotion analysis failed: {e}")
        return []

def process_all_reviews():
    print("🔄 Loading personal reviews file...")

    try:
        with open(r'C:\Users\ajeet\book-recommender\data\personal_reviews.json', 'r', encoding='utf-8') as file:
            reviews = [json.loads(line) for line in file]
        print(f"✅ Loaded {len(reviews)} personal reviews.\n")
    except FileNotFoundError:
        print("❌ Error: Could not find personal_reviews.json. Please check the file location.")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    enriched_reviews = []

    for review in reviews:
        print(f"📚 Processing: {review['book_title']} by {review['author']}")

        sentiment, sentiment_score = analyze_sentiment(review['review_text'])
        top_emotions = analyze_emotions(review['review_text'])

        print(f"   Sentiment: {sentiment} ({sentiment_score:.2f})")
        print(f"   Top 3 Emotions: {top_emotions}")
        print("-" * 50)

        enriched_review = {
            "book_title": review['book_title'],
            "author": review['author'],
            "review_text": review['review_text'],
            "rating": review['rating'],
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "top_emotions": top_emotions
        }
        enriched_reviews.append(enriched_review)

    # Save enriched data to file
    output_path = r'C:\Users\ajeet\book-recommender\data\enriched_reviews.json'
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(enriched_reviews, outfile, indent=4, ensure_ascii=False)

    print(f"\n✅ Analysis complete! Results saved to {output_path}")

if __name__ == "__main__":
    print("🚀 Starting Sentiment and Emotion Analysis")
    process_all_reviews()



