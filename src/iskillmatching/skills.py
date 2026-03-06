import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# файлы
# Adjusted paths assuming execution from project root
INPUT_CSV = "data/jobs_requirements.csv"
OUTPUT_CSV = "output.csv"
SKILLS_FILE = "skills.csv"
TEXT_COLUMN = "requirement_ru"

THRESHOLD = 0.6
BATCH_SIZE = 64

def main():
    # загрузка навыков
    if not os.path.exists(SKILLS_FILE):
        print(f"Error: {SKILLS_FILE} not found.")
        return

    with open(SKILLS_FILE, encoding="utf-8") as f:
        skills = [s.strip().lower() for s in f if s.strip()]

    # модель
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # embedding для навыков
    skill_embeddings = model.encode(skills, normalize_embeddings=True, batch_size=128)

    # загрузка вакансий
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    results = []

    # разделители candidate phrases
    split_pattern = re.compile(r",|\||/|\+|;")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        # разбиваем каждую строку на candidate phrases
        batch_phrases = []
        index_map = []  # какой текст каждой phrase
        for idx, t in enumerate(batch):
            phrases = [p.strip() for p in split_pattern.split(t) if p.strip()]
            batch_phrases.extend(phrases)
            index_map.extend([idx]*len(phrases))

        # embedding для всех candidate phrases
        embeddings = model.encode(batch_phrases, normalize_embeddings=True)

        # similarity
        sims = cosine_similarity(embeddings, skill_embeddings)

        # собираем навыки для каждой строки
        batch_skills = [set() for _ in range(len(batch))]
        for j, row in enumerate(sims):
            for k, score in enumerate(row):
                if score >= THRESHOLD:
                    batch_skills[index_map[j]].add(skills[k])

        # собираем в список
        results.extend(["|".join(sorted(s)) for s in batch_skills])
        print(f"processed {min(i+BATCH_SIZE,len(texts))}/{len(texts)}")

    df["skills_extracted"] = results
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done")

if __name__ == "__main__":
    main()
