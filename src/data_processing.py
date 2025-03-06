import pandas as pd

def load_personal_reviews():
    # This will later load your personal reviews (you’ll create personal_reviews.json)
    return pd.read_json('../data/personal_reviews.json', lines=True)

def load_public_reviews():
    # Placeholder - you can add public data if needed
    return pd.DataFrame()

def preprocess_review(text):
    # Very basic preprocessing for now
    return text.lower().strip()

if __name__ == "__main__":
    # Example flow - you’ll replace this later with actual processing
    personal_data = load_personal_reviews()
    print(f"Loaded {len(personal_data)} personal reviews.")

