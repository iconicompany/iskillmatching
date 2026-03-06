import re

# Common noise words in CV/Job descriptions that are not actual skills
SKILL_STOP_WORDS = {
    "знание", "знания", "умение", "навык", "навыки", "опыт", "понимание", 
    "владение", "проведение", "организация", "наличие", "практический", 
    "практического", "сфера", "сфере", "область", "области", "использование", 
    "использования", "создание", "создания", "проектирование", "разработка", 
    "разработки", "поддержка", "поддержки", "сопровождение", "сопровождения",
    "уверенный", "уверенное", "хорошее", "отличное", "базовое", "глубокое",
    "работа", "работы", "команда", "команде", "командах", "проект", "проектов",
    "продукт", "продуктов", "система", "системы", "системами", "платформа",
    "платформе", "технология", "технологии", "технологиями", "инструмент",
    "инструменты", "инструментами", "библиотека", "библиотеки", "библиотеками",
    "фреймворк", "фреймворки", "фреймворками", "язык", "языка", "языками",
    "уровень", "уровня", "уровне", "часть", "части", "блок", "блока", "блоков",
    "решение", "решения", "решениями", "задача", "задачи", "задачами",
    "требование", "требования", "требованиями", "заказчик", "заказчика",
    "контроль", "контроля", "учет", "учета", "бухгалтерский", "налоговый",
    "финансовый", "корпоративный"
}

def clean_skill(skill):
    """Cleans a single skill string from noise and artifacts."""
    if not skill:
        return ""
        
    # Remove BERT subword artifacts
    skill = skill.replace("##", "")
    
    # Remove common punctuation and noise chars at start/end
    skill = re.sub(r"^[ \-.:,()\"'#+]+", "", skill)
    skill = re.sub(r"[ \-.:,()\"'#+]+$", "", skill)
    
    # Lowercase for comparison with stop words
    skill_lower = skill.lower()
    
    # Split by spaces and filter out stop words
    words = skill_lower.split()
    filtered_words = [w for w in words if w not in SKILL_STOP_WORDS and len(w) > 1]
    
    # If after filtering we have nothing, return empty
    if not filtered_words:
        return ""
        
    # Reconstruct the skill (keeping original casing if needed, but usually skills are lowercase here)
    return " ".join(filtered_words)

def combine_and_clean(ner_piped, spacy_piped):
    """Combines skills from two sources, cleans them, and returns a unique piped string."""
    # Split piped strings into sets
    ner_set = set(ner_piped.split("|")) if ner_piped else set()
    spacy_set = set(spacy_piped.split("|")) if spacy_piped else set()
    
    # Merge and clean
    combined_set = ner_set.union(spacy_set)
    cleaned_skills = []
    seen = set()
    
    for s in combined_set:
        cleaned = clean_skill(s)
        if cleaned and cleaned not in seen:
            cleaned_skills.append(cleaned)
            seen.add(cleaned)
            
    # Sort for consistency
    cleaned_skills.sort()
    
    return "|".join(cleaned_skills)
