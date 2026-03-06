import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
import os

def load_skills_list(file_path):
    """Loads skills from CSV and returns a list of unique, non-empty strings."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    
    try:
        # Try robust reading
        df = pd.read_csv(file_path, on_bad_lines='skip', quotechar='"')
        skill_column = df.columns[0]
        skills = df[skill_column].dropna().unique().tolist()
    except Exception as e:
        print(f"Error reading with pandas: {e}. Falling back to line-by-line reading.")
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f, None)
            skills = [line.strip() for line in f if line.strip()]
            
    return [str(s).strip() for s in skills if str(s).strip()]

def get_spacy_matcher(nlp, skills_list):
    """Initializes spaCy PhraseMatcher with the given skills list."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_list]
    matcher.add("SKILL", patterns)
    return matcher

def extract_spacy_skills(texts, nlp, matcher):
    """Processes a list of texts using nlp.pipe and returns a list of piped skill strings."""
    piped_skills = []
    
    for doc in nlp.pipe(texts):
        matches = matcher(doc)
        seen_skills = set()
        extracted_skills = []
        for match_id, start, end in matches:
            span = doc[start:end]
            skill_name = span.text
            if skill_name.lower() not in seen_skills:
                extracted_skills.append(skill_name)
                seen_skills.add(skill_name.lower())
        piped_skills.append("|".join(extracted_skills))
        
    return piped_skills
