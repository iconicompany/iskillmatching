import pandas as pd
import spacy
import os
import warnings
from .ner_utils import get_ner_extractor, extract_ner_skills
from .spacy_utils import load_skills_list, get_spacy_matcher, extract_spacy_skills
from .clean_utils import combine_and_clean
from .normalize_utils import SkillNormalizer

import argparse

# Configuration (defaults)
TEXT_COLUMN = "requirement_ru"
BATCH_SIZE = 32

def extract_fallback_skills(df):
    """Extracts unique skills from 'skills' and 'stack' columns of the dataframe."""
    fallback_skills = set()
    for col in ["skills", "stack"]:
        if col in df.columns:
            # skills and stack are often piped strings
            series = df[col].fillna("").astype(str)
            for val in series:
                if val:
                    parts = [p.strip().lower() for p in val.split("|") if p.strip()]
                    fallback_skills.update(parts)
    return sorted(list(fallback_skills))

def main():
    parser = argparse.ArgumentParser(description="Extract and normalize skills from CSV.")
    parser.add_argument("-i", "--input", default="data/jobs_requirements.csv", help="Input CSV file path")
    parser.add_argument("-o", "--output", default="data/jobs_requirements_ner.csv", help="Output CSV file path")
    parser.add_argument("-s", "--skills", default="data/skills.csv", help="Skills dictionary CSV file path (optional)")
    parser.add_argument("-l", "--limit", type=int, default=100, help="Number of rows to process (default 100, -1 for no limit)")
    args = parser.parse_args()

    # 1. Read input data
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    df = pd.read_csv(args.input)
    
    # 2. Determine skills list
    if args.skills and os.path.exists(args.skills):
        print(f"Loading skills dictionary from {args.skills}...")
        skills_list = load_skills_list(args.skills)
        normalization_data = args.skills
    else:
        print("No skills dictionary provided or file not found. Extracting fallback skills from input columns...")
        skills_list = extract_fallback_skills(df)
        normalization_data = skills_list
        print(f"Found {len(skills_list)} unique fallback skills.")

    # 3. Apply limit to processing
    if args.limit != -1:
        print(f"Limiting processing to the first {args.limit} rows.")
        df = df.head(args.limit).copy()
    else:
        print("Processing all rows (no limit).")

    # 4. Initialize models
    print("Loading models...")
    ner_extractor = get_ner_extractor()
    
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Model 'en_core_web_lg' not found. Please run: python -m spacy download en_core_web_lg")
        return
        
    matcher = get_spacy_matcher(nlp, skills_list)
    
    # Initialize SkillNormalizer with the chosen model
    modelName = "anass1209/resume-job-matcher-all-MiniLM-L6-v2"
    normalizer = SkillNormalizer(normalization_data, model_name=modelName, threshold=0.8)

    # 5. Prepare for processing
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    total = len(texts)
    
    ner_results = []
    spacy_results = []
    combined_results = []
    normalized_results = []

    print(f"Processing {total} rows from {args.input}...")

    # 6. Batch processing
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        
        # NER Extraction
        ner_batch = extract_ner_skills(batch, ner_extractor)
        ner_results.extend(ner_batch)
        
        # spaCy Extraction
        spacy_batch = extract_spacy_skills(batch, nlp, matcher)
        spacy_results.extend(spacy_batch)
        
        # Combined and Cleaned
        current_combined = []
        for n, s in zip(ner_batch, spacy_batch):
            current_combined.append(combine_and_clean(n, s))
        combined_results.extend(current_combined)
        
        # Normalization
        normalized_batch = normalizer.normalize_batch(current_combined)
        normalized_results.extend(normalized_batch)
        
        print(f"Processed {min(i + BATCH_SIZE, total)}/{total}")

    # 7. Save results
    df["ner"] = ner_results
    df["spacy"] = spacy_results
    df["combined"] = combined_results
    df["normalized"] = normalized_results
    
    df.to_csv(args.output, index=False)
    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()
