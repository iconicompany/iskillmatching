import pandas as pd
import os
import warnings
from .ner_utils import get_ner_extractor, extract_ner_skills

warnings.filterwarnings("ignore")

INPUT_CSV = "data/jobs_requirements.csv"
OUTPUT_CSV = "output_with_ner.csv"
TEXT_COLUMN = "requirement_ru"
BATCH_SIZE = 32

def main():
    print("Loading NER model...")
    ner_extractor = get_ner_extractor()
    
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
        batch_results = extract_ner_skills(batch, ner_extractor)
        results.extend(batch_results)
        print(f"Processed {min(i + BATCH_SIZE, total)}/{total}")

    df["ner"] = results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
