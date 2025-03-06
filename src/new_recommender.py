"""
Book Recommendation System with Emotion & Keyword Matching
Version: 1.0
Author: Pranamya Jain
Date: March 2025
Description:
This script recommends books to the user based on their previously read books, using a combination of:
- Open Library API (for fetching book data)
- DistilBERT Emotion Analysis (to match books emotionally)
- KeyBERT Keyword Extraction (for topical similarity)

Key Features:
- Multi-category search across self-help, psychology, philosophy, and personal development.
- Same-author fallback if no good matches are found.
- Real-time progress bars for better user experience.
- Full logging to 'recommendation_log.txt' for debugging and analysis.
- Option to re-run with a different seed book without restarting.
- Final recommendations saved to 'recommended_books.json'.
"""

# Required Libraries
import json
import os
import requests
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import euclidean_distances
from keybert import KeyBERT
import re
from tqdm import tqdm
import logging

# Setup Logging
logging.basicConfig(
    filename='book_recommender.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message):
    logging.info(message)




# Ensure reproducibility
np.random.seed(42)

# Comprehensive Keyword Set
RELEVANT_KEYWORDS = {
    'self-improvement', 'growth', 'potential', 'mindset', 'discipline',
    'motivation', 'resilience', 'personal transformation', 'self-mastery',
    'psychology', 'emotional intelligence', 'subconscious', 'cognitive skills',
    'mental frameworks', 'perception', 'self-awareness', 'human nature',
    'skill', 'mastery', 'expertise', 'learning', 'practice', 'continuous improvement',
    'deliberate practice', 'performance', 'talent development',
    'communication', 'persuasion', 'charisma', 'social skills', 'influence',
    'negotiation', 'relationship building', 'emotional connection',
    'strategy', 'goal setting', 'productivity', 'success principles', 'decision making',
    'personal philosophy', 'life design', 'overcoming obstacles',
    'emotions', 'emotional control', 'self-regulation', 'mental toughness',
    'confidence', 'inner strength', 'overcoming fear', 'ego management',
    'seduction', 'art of living', 'minimalism', 'stoicism', 'personal power',
    'creativity', 'entrepreneurship', 'life lessons'
    # Self-Improvement & Personal Growth
    'self-improvement', 'personal growth', 'mindset', 'discipline', 'resilience',
    'self-discipline', 'habit building', 'goal setting', 'life planning', 'motivation',
    'time management', 'productivity', 'focus', 'deep work', 'self-awareness',
    'personal transformation', 'growth mindset', 'procrastination', 'limiting beliefs',
    'goal achievement', 'action-taking', 'life design', 'habit stacking', 'morning routine',
    'evening routine', 'journaling', 'visualization', 'success habits', 'habit loops',
    'self-optimization', 'delayed gratification', 'positive habits', 'habit triggers',
    'success rituals', 'personal breakthroughs', 'habit tracking', 'behavioral change',
    'intentional living', 'personal accountability', 'mindset mastery', 'peak performance',
    'habit systems', 'life blueprint', 'personal effectiveness', 'micro habits', 'value alignment',

    # Psychology & Emotional Intelligence
    'psychology', 'emotional intelligence', 'cognitive biases', 'cognitive psychology',
    'subconscious', 'behavioral psychology', 'self-awareness', 'personality psychology',
    'decision heuristics', 'cognitive distortions', 'emotional triggers', 'self-esteem',
    'stress management', 'mindfulness', 'neuroscience', 'brain training', 'emotional agility',
    'positive psychology', 'inner critic', 'mental health', 'rumination', 'self-sabotage',
    'impulse control', 'inner dialogue', 'self-validation', 'thought reframing',
    'emotional healing', 'social cognition', 'psychological resilience', 'neuroplasticity',
    'emotional anchors', 'subconscious programming', 'mental habits', 'behavioral patterns',
    'emotional granularity', 'self-talk', 'mental clarity', 'inner narrative', 'mood regulation',
    'emotional reflexes', 'cognitive flexibility', 'decision fatigue', 'meta-cognition',
    'perception management', 'fear extinction', 'emotional literacy', 'neuro-associations',
    'anxiety management', 'self-soothing', 'inner resourcefulness', 'thought patterns',
    'emotional resilience', 'psychological flexibility', 'self-compassion', 'cognitive training',

    # Skill Acquisition & Mastery
    'skill acquisition', 'deliberate practice', 'learning agility', 'mastery', 'skill development',
    'deep learning', 'competency development', 'learning projects', 'competence mapping',
    'knowledge scaffolding', 'self-directed learning', 'adaptive learning', 'upskilling',
    'performance mastery', 'competency frameworks', 'learning habits', 'knowledge mastery',
    'practice techniques', 'reflective learning', 'learning checklists', 'progress tracking',
    'learning journals', 'personal learning plans', 'learning mindsets', 'knowledge integration',
    'skill mastery systems', 'learning optimization', 'learning feedback', 'competency audits',
    'learning blueprints', 'competence roadmaps', 'learning assessments', 'competence tracking',
    'learning cycle reviews', 'skill strengthening', 'learning experiments', 'learning milestones',
    'learning sprints', 'practice deep dives', 'learning review loops', 'microlearning',

    # Communication & Interpersonal Skills
    'communication', 'persuasion', 'influence', 'negotiation', 'conflict resolution',
    'public speaking', 'presentation skills', 'active listening', 'storytelling', 'rapport building',
    'social intelligence', 'nonverbal communication', 'body language', 'conversational skills',
    'empathy', 'assertiveness', 'difficult conversations', 'effective communication', 'charisma',
    'relationship building', 'personal branding', 'networking', 'team communication',
    'persuasion strategies', 'collaborative communication', 'conflict management',
    'feedback delivery', 'personal storytelling', 'trust building', 'engagement skills',
    'communication frameworks', 'pitching ideas', 'persuasive language', 'authentic communication',
    'small talk', 'relationship intelligence', 'social calibration', 'relationship nurturing',
    'verbal communication', 'written communication', 'narrative building', 'social fluency',
    'team dynamics', 'empathy-based communication', 'cross-cultural communication',

    # Professional & Personal Strategy
    'strategy', 'goal setting', 'decision making', 'personal philosophy', 'critical thinking',
    'life planning', 'tactical thinking', 'problem solving', 'success principles', 'decision frameworks',
    'competitive advantage', 'scenario planning', 'personal vision', 'future planning',
    'mental models', 'strategic thinking', 'decision processes', 'goal frameworks', 'life architecture',
    'priority management', 'performance measurement', 'goal alignment', 'personal KPIs',
    'outcome-based thinking', 'systems thinking', 'life optimization', 'strategic prioritization',
    'personal success strategies', 'time optimization', 'decision mapping', 'personal SWOT analysis',
    'goal recalibration', 'long-term visioning', 'focus alignment', 'decision confidence',
    'adaptive planning', 'goal flexibility', 'risk management', 'success roadmaps', 'opportunity spotting',
    'habit strategy', 'personal leverage', 'life planning frameworks', 'adaptive strategies',
    'self-strategy alignment', 'decision audits', 'big picture thinking', 'personal governance',

    # Emotional Management
    'emotional control', 'self-regulation', 'emotional resilience', 'inner strength', 'overcoming fear',
    'anger management', 'stress reduction', 'calmness', 'emotional mastery', 'handling criticism',
    'fear management', 'managing expectations', 'self-control', 'emotional flexibility',
    'coping strategies', 'frustration tolerance', 'emotional detox', 'emotional independence',
    'managing uncertainty', 'emotional stability', 'inner calm', 'emotional detachment', 'emotional anchoring',
    'reaction control', 'stress inoculation', 'fearless mindset', 'emotion-focused coping',
    'resilience training', 'stress adaptation', 'emotional renewal', 'emotional self-sufficiency',
    'worry management', 'coping skills', 'emotional balance', 'emotional toughness', 'crisis resilience',
    'emotional fitness', 'emotional recalibration', 'handling rejection', 'emotional wellbeing',
    'self-centering', 'emotional de-escalation', 'inner balance', 'fear processing', 'resilience habits',

    # Specific Themes
    'seduction', 'art of living', 'minimalism', 'stoicism', 'personal power', 'creativity',
    'entrepreneurship', 'life lessons', 'life philosophy', 'mindfulness techniques', 'zen habits',
    'spiritual growth', 'personal fulfillment', 'purpose discovery', 'meditation', 'value discovery',
    'mind-body connection', 'self-acceptance', 'existential thinking', 'inner transformation',
    'life experiments', 'voluntary discomfort', 'life optimization', 'self-reliance', 'personal awakening',
    'philosophical inquiry', 'holistic growth', 'life artistry', 'personal legacy', 'life curation',
    'inner alignment', 'self-actualization', 'meaningful living', 'personal philosophy crafting',
    'wisdom cultivation', 'core values', 'life congruence', 'spiritual alignment', 'inner dialogue mastery'
}

KEYWORD_WEIGHT = 0.4
EMOTION_WEIGHT = 0.3
DESCRIPTION_WEIGHT = 0.3

def safe_load_json(file_path):
    """
    Safely loads a JSON file from the given path.
    Handles file not found and JSON decoding errors.
    Returns an empty list if loading fails.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def fetch_book_details(work_key):
    
    try:
        url = f"https://openlibrary.org{work_key}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        description = data.get('description', '')

        if isinstance(description, dict):
            description = description.get('value', '')

        return str(description)[:1500] or "No description available."
    except (requests.RequestException, ValueError):
        return "No description available."

def fetch_books_from_all_categories(categories, limit=50):
    all_books = []
    for subject in categories:
        print(f"\n🔍 Searching for books in '{subject}' category...")
        url = f"https://openlibrary.org/subjects/{subject}.json?limit={limit}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()
            for entry in data.get('works', []):
                work_key = entry['key']
                description = fetch_book_details(work_key)

                if len(description.strip()) > 50:
                    all_books.append({
                        "title": entry['title'],
                        "authors": [author.get('name', 'Unknown') for author in entry.get('authors', [])],
                        "description": description,
                        "link": f"https://openlibrary.org{work_key}"
                    })
        except requests.RequestException:
            print(f"❌ Failed to fetch books for {subject}")

    return all_books
def fetch_books_by_author(author_name, limit=10):
    """Fetch other books by the same author from Open Library."""
    try:
        query = author_name.replace(" ", "+")
        url = f"https://openlibrary.org/search.json?author={query}&limit={limit}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()
        books = []
        for doc in data.get('docs', []):
            description = fetch_book_details(doc['key'])  # Fetch details using the work key (same as your existing logic)
            books.append({
                "title": doc.get('title', 'Untitled'),
                "authors": [author_name],
                "description": description,
                "openlibrary_url": f"https://openlibrary.org{doc['key']}"
            })
        return books

    except Exception as e:
        print(f"❌ Failed to fetch books by author '{author_name}': {e}")
        return []

def is_relevant_book(book, seed_keywords, already_read):
    description = book['description'].lower()
    title = book['title'].lower()

    if title in already_read:
        return False

    return (
        len(description) > 50 and
        any(keyword in description or keyword in title for keyword in RELEVANT_KEYWORDS) and
        any(keyword in description or keyword in title for keyword in seed_keywords)
    )

def extract_keywords(text):
    try:
        keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=15
        )
        return {kw[0].lower() for kw in keywords if len(kw[0].split()) <= 3}
    except Exception:
        return set()

def analyze_emotions(text, emotion_pipeline):
    try:
        result = emotion_pipeline(text, top_k=3)
        if isinstance(result[0], list):
            result = result[0]

        return {emo['label']: emo['score'] for emo in result}
    except Exception:
        return {}

def vectorize_emotions(emotions, all_emotions):
    return np.array([emotions.get(emotion, 0.0) for emotion in all_emotions])

def calculate_description_relevance(description, seed_keywords):
    description = description.lower()
    return min(sum(description.count(keyword) * 1.5 for keyword in seed_keywords) / len(seed_keywords), 1.0)

def summarize_description(description, word_limit=100):
    words = re.split(r'\s+', description)
    return ' '.join(words[:word_limit]) + ('...' if len(words) > word_limit else '')


def handle_fallback_recommendation(seed_book, top_n=5):
    """Fallback to recommending books by the same author if no good matches found."""
    # Check the actual structure of seed_book and extract author name correctly

    if 'author' in seed_book:
        author_name = seed_book['author']
    elif 'book_author' in seed_book:
        author_name = seed_book['book_author']
    else:
        print("❌ Could not determine author for fallback recommendations")
        return
    print("\n" + "=" * 80)

    print(f"📣 RECOMMENDATION NOTICE: We couldn't find books with strong content similarity to '{seed_book.get('book_title', 'your selected book')}'.")

    print(f"\n🔄So fetching books by {author_name} instead...")
    
    print("=" * 80 + "\n")


    author_books = fetch_books_by_author(author_name, limit=top_n)

    if not author_books:
        print(f"❌ No books found for author: {author_name}")
        return

    print(f"\n📚 Books by {author_name}:\n")
    for idx, book in enumerate(author_books, 1):
        print(f"{idx}. 🔹 {book['title']}")
        print(f"   📖 {book['description'][:300]}{'...' if len(book['description']) > 300 else ''}")
        print(f"   🔗 View Online: {book['openlibrary_url']}")
        print("-" * 60)


def recommend_books(seed_emotions, seed_keywords, candidate_books, emotion_pipeline, all_emotions, already_read, seed_book, top_n=7):
    seed_vector = vectorize_emotions(seed_emotions, all_emotions)
    fallback_triggered = False  # ✅ Initialize here

    recommendations = []

    for book in tqdm(candidate_books, desc="🔎 Processing Books", unit="book"):

        if not is_relevant_book(book, seed_keywords, already_read):
            continue

        emotions = analyze_emotions(book['description'] or book['title'], emotion_pipeline)
        emotion_vector = vectorize_emotions(emotions, all_emotions)

        book_keywords = extract_keywords(book['description'])
        keyword_overlap = len(seed_keywords.intersection(book_keywords))

        emotion_distance = np.linalg.norm(seed_vector - emotion_vector)
        emotion_similarity = 1 / (1 + emotion_distance)

        keyword_score = keyword_overlap / max(len(seed_keywords), 1)
        description_relevance = calculate_description_relevance(book['description'], seed_keywords)

        total_score = (
            KEYWORD_WEIGHT * keyword_score +
            EMOTION_WEIGHT * emotion_similarity +
            DESCRIPTION_WEIGHT * description_relevance
        )

        recommendations.append({
            'title': book['title'],
            'authors': book['authors'],
            'description': summarize_description(book['description']),
            'link': book['link'],
            'score': total_score,
            'keyword_score': keyword_score,
            'emotion_score': emotion_similarity,
            'description_relevance': description_relevance
        })

    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]

    # Trigger fallback to author's books if no strong matches found
    strong_matches = [rec for rec in recommendations if rec['score'] >= 0.4]

    if not strong_matches:
        fallback_triggered = True  # ✅ Set this if fallback is triggered
        handle_fallback_recommendation(seed_book)
        return  # Skip saving irrelevant recommendations

    with open('recommended_books.json', 'w', encoding='utf-8') as file:
        json.dump(recommendations, file, indent=4)

    print("\n📚 Precision Book Recommendations (also saved to 'recommended_books.json'):\n")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. 🔹 {rec['title']}")
        print(f"   By: {', '.join(rec['authors'])}")
        print(f"   Overall Relevance: {rec['score']:.3f} (calculated as follows):")
        print(f"      - Keyword Match: {rec['keyword_score']:.3f} × {KEYWORD_WEIGHT} = {rec['keyword_score'] * KEYWORD_WEIGHT:.3f}")
        print(f"      - Emotional Similarity: {rec['emotion_score']:.3f} × {EMOTION_WEIGHT} = {rec['emotion_score'] * EMOTION_WEIGHT:.3f}")
        print(f"      - Content Relevance: {rec['description_relevance']:.3f} × {DESCRIPTION_WEIGHT} = {rec['description_relevance'] * DESCRIPTION_WEIGHT:.3f}")
        print(f"   📖 {rec['description']}")
        print(f"   🔗 View Online: {rec['link']}")
        print("-" * 60)

    # Calculate stats and print summary
    total_books = len(candidate_books)
    filtered_out = total_books - len([book for book in candidate_books if is_relevant_book(book, seed_keywords, already_read)])

    # ✅ Now pass `fallback_triggered` to print_summary()
    print_summary(total_books, filtered_out, len(recommendations), fallback_used=fallback_triggered)

def print_summary(total_books, filtered_out, recommended_count, fallback_used):
    print("\n📊 Recommendation Process Summary:")
    print(f"   🔎 Books Fetched: {total_books}")
    print(f"   🚫 Books Filtered Out (Irrelevant/Duplicates/Already Read): {filtered_out}")
    print(f"   ✅ Books Analyzed: {total_books - filtered_out}")
    print(f"   ⭐ Final Recommendations: {recommended_count}")
    if fallback_used:
        print(f"   🔄 Same-Author Fallback Triggered: Yes")
    else:
        print(f"   🔄 Same-Author Fallback Triggered: No")
    print("-" * 60)
    print("\n📚 Happy Reading!")




def main():
    print("\n🚀 Welcome to The Intelligent Book Recommender!")
    print("📚 Truly Powered by Open Library & Emotion Analysis!")
    print("==============================================\n")


    log("📖 New recommendation process started.")

    BASE_PATH = r'C:\Users\ajeet\book-recommender'
    REVIEWS_PATH = os.path.join(BASE_PATH, 'data', 'enriched_reviews.json')

    emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    global kw_model
    kw_model = KeyBERT()

    personal_books = safe_load_json(REVIEWS_PATH)
    if not personal_books:
        log("❌ No books found in personal collection.")
        print("❌ No books found in your collection.")
        return

    log(f"✅ Loaded {len(personal_books)} books from personal collection.")

    print("\n📚 Your Books:\n")
    for idx, book in enumerate(personal_books, 1):
        print(f"{idx}. {book['book_title']}")

    log("Displayed list of personal books.")

    already_read = {book['book_title'].lower() for book in personal_books}

    while True:  # Allow repeated book recommendations without restarting
        seed_title = input("\nEnter the title of a book you've read (exactly as shown above): ").strip()
        seed_book = next((book for book in personal_books if book['book_title'].strip().lower() == seed_title.strip().lower()), None)


        if not seed_book:
            log(f"❌ Book '{seed_title}' not found in personal collection.")
            print(f"❌ Book '{seed_title}' not found. Please try again.")
            continue  # Let user try again without restarting

        log(f"✅ Selected seed book: {seed_book['book_title']}")

        seed_emotions = dict(seed_book['top_emotions'])
        seed_keywords = extract_keywords(seed_book['review_text'])

        # Let the user choose categories to search
        available_categories = ['self_help', 'psychology', 'philosophy', 'personal_development']
        print("\n🔍 Available Categories to Search:")
        for idx, cat in enumerate(available_categories, 1):
            print(f"   {idx}. {cat}")

        selected_indexes = input("\nEnter the numbers of the categories you want to search (comma-separated, e.g., 1,3): ").strip()
        selected_categories = [available_categories[int(i) - 1] for i in selected_indexes.split(',') if i.strip().isdigit()]

        if not selected_categories:
            print("❌ No valid categories selected. Exiting.")
            return

        log(f"🔎 User selected categories: {', '.join(selected_categories)}")

        candidate_books = fetch_books_from_all_categories(selected_categories, limit=50)

        all_emotions = ["joy", "curiosity", "love", "anger", "fear", "sadness"]

        log("🔎 Starting recommendation process.")
        recommend_books(seed_emotions, seed_keywords, candidate_books, emotion_pipeline, all_emotions, already_read, seed_book)
        log("✅ Recommendation process completed.")

        # Ask if user wants to run again
        run_again = input("\n🔄 Do you want to search recommendations for another book? (yes/no): ").strip().lower()
        if run_again != 'yes':
            log("🚪 User chose to exit after recommendation process.")
            print("\n📚 Byeeee! Exiting the system.")
            break
    print("\n👋 Thank you for using the Intelligent Book Recommender! A Very Happy Reading! 📖✨")


if __name__ == "__main__":
    main()
