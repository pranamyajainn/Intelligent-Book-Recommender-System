# Book Recommender System

📚 An intelligent **book recommendation system** that recommends books based on:

- Emotional similarity with books you’ve already read.
- Keyword and content relevance.
- Multi-category search across self-help, psychology, philosophy, and personal development.
- Optional fallback to recommend books from the **same author** if no strong matches are found.

---

## 🚀 Features
✅ **Emotion Analysis** — Understands emotional tone using NLP.<br>
✅ **Keyword Matching** — Matches key concepts with new books.<br>
✅ **Progress Bar** — Shows real-time book processing progress.<br>
✅ **Multi-Category Search** — Searches across 4 categories for maximum coverage.<br>
✅ **Same Author Fallback** — If no strong matches found, suggests books by the same author.<br>
✅ **Logging** — Tracks process for debugging.<br>
✅ **Case Insensitive Search** — No more issues with title case mismatches.<br>

---
## 🌟 What Makes This Project Unique?

This project goes beyond traditional **book recommendation systems** in several ways:

1. **Emotion-Based Matching:** 
   It doesn’t just match books by category or popularity — it analyzes the **emotional tone** of books you love and finds new books that evoke **similar emotions**.

2. **Keyword Relevance Scoring:** 
   By leveraging **KeyBERT**, it extracts and compares **key themes and concepts** from book descriptions and reviews, ensuring topic relevance.

3. **Smart Fallback System:** 
   If no good matches are found, the system automatically falls back to **recommending other books by the same author** — blending personalization with practical suggestions.

4. **Multi-Category Search:** 
   Instead of limiting recommendations to a single genre, it pulls books from **self-help, psychology, philosophy, and personal development**, giving you a richer and more diverse selection.

5. **Advanced Filtering:** 
   It actively removes books you’ve already read, irrelevant books, and low-quality books with missing or poor descriptions.

6. **Progress Tracking & Summaries:** 
   The system provides a **step-by-step progress bar** and a final **recommendation summary** — giving you full transparency into how books were selected.

7. **Flexible Deployment Options:** 
   Whether you want to run it locally, **convert it to a user-friendly web app (Streamlit)**, or package it as a **standalone .exe** — this project supports all these options.

8. **Future-Ready:** 
   With planned features like **automated monthly recommendations sent via email**, this project can evolve into a **personalized book concierge**.

This combination of **emotion analysis, keyword extraction, smart fallback logic, and cross-category search** makes this project far more than a typical book recommender.

---

## 🔧 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Navigate to the project folder:
    ```bash
    cd your-repo-name
    ```
3. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # For Linux/Mac
    .\venv\Scripts\activate    # For Windows
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Usage

1. Run the main script:
    ```bash
    python new_recommendation.py
    ```
2. Follow on-screen instructions to select a book you’ve read and receive personalized recommendations.

---

## 🧰 Dependencies
- `transformers`
- `torch`
- `scikit-learn`
- `keybert`
- `langdetect`
- `requests`
- `tqdm`

All dependencies are listed in `requirements.txt`.

---
## 📂 Project Structure

book-recommender/
│
├── data/
│   ├── enriched_reviews.json
│   └── recommended_books.json
│
├── src/
│   ├── new_recommender.py        # Main script
│   ├── data_processing.py        # (optional — pre-processing if needed)
│   ├── sentiment_analysis.py     # (optional — if you separate sentiment logic)
│
├── logs/
│   └── book_recommender.log       # Log file
│
├── .gitignore
├── requirements.txt
├── README.md
└── environment.yml                # (optional — if using Conda)


---

## 🏗️ Future Enhancements (Optional)
- Convert to Streamlit Web App.
- Package as `.exe` for easy sharing.
- Scheduled automation (fetch books + recommend monthly).

---

## 📝 Author
- Created by **Pranamya Jain**
- Reach me on [LinkedIn](https://www.linkedin.com/in/pranamya-jain1/)

---

## 📜 License
MIT License — Free to use, modify, and share.

---
