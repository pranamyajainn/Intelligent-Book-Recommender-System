import json
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the enriched book data
def load_data():
    try:
        with open(r'C:\Users\ajeet\book-recommender\data\enriched_reviews.json', 'r', encoding='utf-8') as file:
            books = json.load(file)
        print(f"✅ Loaded {len(books)} books from enriched_reviews.json\n")
        return books
    except FileNotFoundError:
        print("❌ Error: Could not find enriched_reviews.json. Run sentiment_analysis.py first.")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []

# Convert emotions to a fixed-size vector
def get_emotion_vector(book, all_emotions):
    """ Returns a vectorized representation of emotions for a given book. """
    emotion_dict = {emotion: 0.0 for emotion in all_emotions}  # Initialize all emotions with 0

    for emotion, score in book.get("top_emotions", []):
        emotion_dict[emotion] = score  # Assign detected emotion scores

    return np.array(list(emotion_dict.values()))

# Compute similarity using Euclidean Distance converted into Similarity
def compute_similarity(book_vectors):
    """ Compute pairwise similarity (inverse of Euclidean Distance). """
    distances = euclidean_distances(book_vectors, book_vectors)

    # Convert distance to similarity (lower distance = higher similarity)
    similarity = 1 / (1 + distances)  # Invert to make 0 distance = highest similarity (1.0)

    return similarity

# Recommend books with most similar emotional profiles
def recommend_books(book_title, books, similarity_matrix, book_titles, top_n=5):
    if book_title not in book_titles:
        print(f"❌ Error: '{book_title}' not found in the dataset.")
        return []

    book_idx = book_titles.index(book_title)
    similarity_scores = similarity_matrix[book_idx]

    # Get top N most similar books (excluding the book itself)
    similar_books = sorted(
        [(book_titles[i], similarity_scores[i]) for i in range(len(books)) if i != book_idx],
        key=lambda x: -x[1]  # Sort by similarity (higher is better)
    )[:top_n]

    print(f"\n📚 **Top {top_n} Emotionally Similar Books to '{book_title}'**:\n")
    for title, score in similar_books:
        print(f"🔹 {title} (Similarity Score: {score:.3f})")

    return similar_books

def main():
    # Step 1: Load book data
    books = load_data()
    if not books:
        return

    # Step 2: Extract all unique emotions across all books
    all_emotions = set()
    for book in books:
        for emotion, _ in book.get("top_emotions", []):
            all_emotions.add(emotion)

    all_emotions = sorted(list(all_emotions))  # Keep emotions consistent across all books

    # Step 3: Convert all books into fixed-size emotion vectors
    book_vectors = np.array([get_emotion_vector(book, all_emotions) for book in books])

    # Step 4: Compute similarity matrix using Euclidean-based similarity
    similarity_matrix = compute_similarity(book_vectors)

    # Step 5: Ask user for a book and recommend similar books
    book_titles = [book["book_title"] for book in books]

    print("\n📚 Available Books in the Dataset:\n")
    for idx, title in enumerate(book_titles, 1):
        print(f"{idx}. {title}")

    book_title = input("\nEnter the name of a book you've read: ").strip()

    recommend_books(book_title, books, similarity_matrix, book_titles)

if __name__ == "__main__":
    print("🚀 Starting Book Recommendation System...\n")
    main()


