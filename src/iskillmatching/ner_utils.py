from transformers import pipeline

def get_ner_extractor(model_name="dondosss/rubert-finetuned-ner"):
    """Initializes the Transformers NER pipeline."""
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple"
    )

def extract_ner_skills(texts, ner_extractor):
    """Processes a list of texts and returns a list of piped skill strings."""
    if not texts:
        return []
        
    results = ner_extractor(texts)
    
    # Handle single result (if texts was a string, though we expect a list)
    if isinstance(results, dict):
        results = [results]
        
    piped_skills = []
    for ents in results:
        skills = list(set([
            e["word"].lower()
            for e in ents
        ]))
        piped_skills.append("|".join(skills))
        
    return piped_skills
