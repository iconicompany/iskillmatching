import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .spacy_utils import load_skills_list

class SkillNormalizer:
    def __init__(self, skills_data, model_name, threshold=0.6):
        """
        skills_data: can be a path to a CSV or a list of skills.
        """
        print(f"Initializing SkillNormalizer with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        
        if isinstance(skills_data, str):
            print(f"Loading and embedding skills from {skills_data}...")
            self.skills = load_skills_list(skills_data)
        else:
            print("Using provided skills list for embedding...")
            self.skills = skills_data

        if not self.skills:
            print("Warning: Skills list is empty!")
            self.skill_embeddings = np.array([])
        else:
            self.skill_embeddings = self.model.encode(
                self.skills, 
                normalize_embeddings=True, 
                batch_size=128,
                show_progress_bar=False
            )

    def normalize_batch(self, piped_skills_batch):
        """
        Takes a list of piped skill strings (e.g., ["skill1|skill2", "skill3"])
        Returns a list of normalized piped skill strings.
        """
        if self.skill_embeddings.size == 0:
            return piped_skills_batch

        results = []
        for piped_str in piped_skills_batch:
            if not piped_str:
                results.append("")
                continue
                
            current_skills = [s.strip() for s in piped_str.split("|") if s.strip()]
            if not current_skills:
                results.append("")
                continue
                
            # Encode current skills
            embeddings = self.model.encode(current_skills, normalize_embeddings=True, show_progress_bar=False)
            
            # Compute similarity
            sims = cosine_similarity(embeddings, self.skill_embeddings)
            
            normalized_set = set()
            for row in sims:
                # Find the best match above threshold
                best_idx = np.argmax(row)
                if row[best_idx] >= self.threshold:
                    normalized_set.add(self.skills[best_idx])
            
            sorted_normalized = sorted(list(normalized_set))
            results.append("|".join(sorted_normalized))
            
        return results
