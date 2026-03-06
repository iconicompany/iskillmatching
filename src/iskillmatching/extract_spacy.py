import pandas as pd
import spacy
import os
import warnings
from .spacy_utils import load_skills_list, get_spacy_matcher, extract_spacy_skills

warnings.filterwarnings("ignore")

INPUT_CSV = "data/jobs_requirements.csv"
OUTPUT_CSV = "output_with_spacy.csv"
SKILLS_CSV = "skills.csv"
TEXT_COLUMN = "requirement_ru"
BATCH_SIZE = 32

def main():
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Model 'en_core_web_lg' not found.")
        return
        
    skills_list = load_skills_list(SKILLS_CSV)
    matcher = get_spacy_matcher(nlp, skills_list)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV).head(1000).copy()
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    total = len(texts)
    results = []

    print(f"Processing {total} rows...")

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_results = extract_spacy_skills(batch, nlp, matcher)
        results.extend(batch_results)
        print(f"Processed {min(i + BATCH_SIZE, total)}/{total}")

    df["spacy"] = results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
