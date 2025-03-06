import os

file_path = r'C:\Users\ajeet\book-recommender\data\personal_reviews.json'

if os.path.exists(file_path):
    print(f"✅ File exists: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for _ in file)
    print(f"✅ File is readable with {line_count} lines.")
else:
    print(f"❌ File does NOT exist: {file_path}")

