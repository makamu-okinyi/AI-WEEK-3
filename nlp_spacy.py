# nlp_spacy.py

import spacy

# Load the small English model from spaCy.
# You may need to download it first by running: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "The new Sony WH-1000XM4 headphones are absolutely fantastic! The noise cancellation is top-notch.",
    "I bought a Samsung Galaxy S21 and was very disappointed. The battery life is terrible.",
    "This Anker PowerCore charger is a lifesaver for traveling. Highly recommended!",
    "The Logitech MX Master 3 mouse stopped working after just two weeks. A complete waste of money."
]

# 1. Perform Named Entity Recognition (NER)
print("--- Named Entity Recognition (NER) ---")
for review in reviews:
    doc = nlp(review)
    print(f"\nReview: '{review}'")
    print("Extracted Entities:")
    found_entities = False
    for ent in doc.ents:
        # We are interested in organizations (ORG) and products (PRODUCT)
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f"  - Entity: '{ent.text}', Label: '{ent.label_}'")
            found_entities = True
    if not found_entities:
        print("  - No relevant entities found.")


# 2. Analyze Sentiment using a Rule-Based Approach
print("\n" + "--- Rule-Based Sentiment Analysis ---")

# Define simple keyword lists
positive_keywords = ["fantastic", "top-notch", "lifesaver", "highly recommended", "great", "love"]
negative_keywords = ["disappointed", "terrible", "stopped working", "waste of money", "bad"]

for review in reviews:
    doc = nlp(review)
    
    # Calculate sentiment score
    pos_score = 0
    neg_score = 0
    
    # We check the lemmatized form of the token to match variations (e.g., 'working' -> 'work')
    for token in doc:
        if token.lemma_.lower() in positive_keywords:
            pos_score += 1
        elif token.lemma_.lower() in negative_keywords:
            neg_score += 1
            
    # Determine final sentiment
    sentiment = "Neutral"
    if pos_score > neg_score:
        sentiment = "Positive"
    elif neg_score > pos_score:
        sentiment = "Negative"

    print(f"\nReview: '{review}'")
    print(f"Sentiment: {sentiment} (Scores: Pos={pos_score}, Neg={neg_score})")