# -*- coding: utf-8 -*-
"""Convert skills.csv (one skill per line) into CoNLL-format NER training data.

The script reads a skills dictionary and generates synthetic training sentences
using context templates.  Each sentence is output in CoNLL token-per-line
format::

    word TAG

where TAG is B-SKILL (beginning of skill), I-SKILL (inside multi-word skill),
or O (non-skill token).  The output files are ready to use with an NER model
training pipeline that accepts CoNLL-format data.

Usage (run from project root)::

    python data/prepare_skills_data.py \
        --skills data/skills.csv \
        --output_dir data/skills \
        --train_ratio 0.7 --valid_ratio 0.15

Then update the model config to point at the generated files and run the
data-preparation cells (tokeniser fit, label-encoder fit, serialisation)
before training.  NUM_CLASSES will be 3: O, B-SKILL, I-SKILL.
"""

import argparse
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Tokenise text by separating punctuation into distinct tokens.

    This mirrors the preprocessing applied during model inference so that
    token boundaries at training time and at inference time are identical.

    Examples::

        >>> tp = TextPreprocessor()
        >>> tp(".Net")
        '. Net'
        >>> tp("C++")
        'C + +'
        >>> tp("Spring Boot")
        'Spring Boot'
        >>> tp("Experience with React is required.")
        'Experience with React is required .'
    """

    def __call__(self, text: str) -> str:
        """Return *text* with punctuation separated from adjacent tokens."""
        # Insert a space before and after every non-word, non-space character.
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        # Collapse runs of whitespace to a single space and strip ends.
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Sentence templates – {skill} is replaced with the actual skill name.
# Multiple templates ensure the model sees each skill in varied contexts.
# ---------------------------------------------------------------------------
SKILL_TEMPLATES = [
    "Experience with {skill} is required.",
    "Looking for a {skill} developer.",
    "Strong knowledge of {skill} is preferred.",
    "Proficiency in {skill} is a must.",
    "The candidate should know {skill}.",
    "{skill} experience is a significant advantage.",
    "Must have solid {skill} skills.",
    "Background in {skill} is required.",
    "We need {skill} expertise on the team.",
    "Skills: {skill}.",
    "{skill} - intermediate level.",
    "Working knowledge of {skill} is expected.",
    "Familiarity with {skill} is a plus.",
    "Expert-level {skill} knowledge required.",
    "Hands-on {skill} project experience preferred.",
    "Ability to work with {skill} in a production environment.",
    "Demonstrated experience using {skill}.",
    "At least 2 years of {skill} experience.",
    "Our stack includes {skill} among other technologies.",
    "You will be working extensively with {skill}.",
]

# Sentences that contain no skills – used as negative (O-only) examples to
# prevent the model from over-labelling every token.
NON_SKILL_SENTENCES = [
    "The meeting will be held on Monday morning.",
    "Please submit your application by Friday.",
    "The team consists of highly experienced professionals.",
    "We offer a competitive salary and great benefits.",
    "The office is located in the city centre.",
    "All applications will be reviewed carefully.",
    "Interviews will be conducted in the following weeks.",
    "The project deadline is approaching rapidly.",
    "Please send your resume to our HR department.",
    "We are a fast-growing company in the technology sector.",
    "The position is open to candidates worldwide.",
    "Remote work is available for this role.",
    "You will report directly to the engineering manager.",
    "The team works in two-week sprints.",
    "We value creativity, collaboration, and continuous learning.",
    "A bachelor's degree in a relevant field is required.",
    "Excellent communication skills are essential.",
    "We are an equal-opportunity employer.",
    "Relocation assistance is available if needed.",
    "This is a full-time permanent position.",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def read_skills(path: str, skip_header: bool = True) -> list[str]:
    """Read skills from *path* (one skill per line).

    Parameters
    ----------
    path:
        Path to the skills file.  The file may optionally begin with a header
        row (e.g. *Название*); pass ``skip_header=True`` to drop it.
    skip_header:
        When ``True`` (default) the first non-empty line is treated as a
        column header and skipped.
    """
    skills = []
    with open(path, encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    start = 1 if skip_header and lines else 0
    for line in lines[start:]:
        skills.append(line)
    return skills


def find_sublist(haystack: list, needle: list) -> int:
    """Return the start index of *needle* inside *haystack*, or -1."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def make_conll_tokens(
    template: str,
    skill: str,
    preprocessor: TextPreprocessor,
) -> list[tuple[str, str]]:
    """Return a list of ``(word, tag)`` pairs for one template + skill.

    The function preprocesses the full sentence with the same
    :class:`TextPreprocessor` used by the model at inference time, then
    locates the skill tokens inside the sentence token list to assign
    *B-SKILL* / *I-SKILL* tags.  All other tokens receive the *O* tag.
    """
    sentence = template.format(skill=skill)

    # Preprocess and tokenise the full sentence.
    all_tokens = preprocessor(sentence).split()

    # Preprocess and tokenise the skill on its own.
    skill_tokens = preprocessor(skill).split()

    if not skill_tokens:
        return [(tok, "O") for tok in all_tokens]

    # Locate skill tokens inside the sentence token list.
    skill_start = find_sublist(all_tokens, skill_tokens)

    if skill_start == -1:
        # Fallback: label all tokens O rather than silently mislabel.
        return [(tok, "O") for tok in all_tokens]

    result = []
    for i, token in enumerate(all_tokens):
        if i == skill_start:
            result.append((token, "B-SKILL"))
        elif skill_start < i < skill_start + len(skill_tokens):
            result.append((token, "I-SKILL"))
        else:
            result.append((token, "O"))

    return result


def generate_examples(
    skills: list[str],
    preprocessor: TextPreprocessor,
    templates: list[str] | None = None,
) -> list[list[tuple[str, str]]]:
    """Generate CoNLL-format examples for every skill × template combination.

    Each example is a list of ``(word, tag)`` pairs representing one sentence.
    """
    if templates is None:
        templates = SKILL_TEMPLATES

    examples: list[list[tuple[str, str]]] = []

    for skill in skills:
        for template in templates:
            tokens = make_conll_tokens(template, skill, preprocessor)
            if tokens:
                examples.append(tokens)

    # Add non-skill (O-only) sentences.
    for sentence in NON_SKILL_SENTENCES:
        tokens = [(tok, "O") for tok in preprocessor(sentence).split()]
        if tokens:
            examples.append(tokens)

    return examples


def write_conll_file(path: str, examples: list[list[tuple[str, str]]]) -> None:
    """Write CoNLL-format data to *path* (creates parent directories)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for sentence in examples:
            for word, tag in sentence:
                fh.write(f"{word} {tag}\n")
            fh.write("\n")


def split_data(
    examples: list,
    train_ratio: float,
    valid_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Shuffle and split *examples* into train / valid / test subsets."""
    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = examples[:n_train]
    valid = examples[n_train : n_train + n_valid]
    test = examples[n_train + n_valid :]

    return train, valid, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare skills NER training data from a skills dictionary."
    )
    parser.add_argument(
        "--skills",
        default="data/skills.csv",
        help="Path to skills file (one skill per line). Default: data/skills.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/skills",
        help="Output directory for train/valid/test CoNLL files. Default: data/skills",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data used for training. Default: 0.7",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Fraction of data used for validation. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Default: 42",
    )
    parser.add_argument(
        "--no-skip-header",
        dest="skip_header",
        action="store_false",
        default=True,
        help="Do not skip the first line of the skills file (default: skip it).",
    )
    args = parser.parse_args()

    if args.train_ratio + args.valid_ratio >= 1.0:
        parser.error("train_ratio + valid_ratio must be less than 1.0")

    preprocessor = TextPreprocessor()

    print(f"Reading skills from {args.skills} ...")
    skills = read_skills(args.skills, skip_header=args.skip_header)
    print(f"  {len(skills)} skills loaded.")

    print("Generating training examples ...")
    examples = generate_examples(skills, preprocessor)
    print(f"  {len(examples)} sentences generated.")

    train, valid, test = split_data(
        examples, args.train_ratio, args.valid_ratio, args.seed
    )

    train_path = os.path.join(args.output_dir, "train.txt")
    valid_path = os.path.join(args.output_dir, "valid.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    write_conll_file(train_path, train)
    write_conll_file(valid_path, valid)
    write_conll_file(test_path, test)

    print(f"\nSplit: train={len(train)}, valid={len(valid)}, test={len(test)}")
    print(f"Files written to {args.output_dir}/\n")

    print("Next steps:")
    print("  1. Update config.json:")
    print(f'     "PATH_TRAIN": "{train_path}",')
    print(f'     "PATH_VALID": "{valid_path}",')
    print(f'     "PATH_TEST":  "{test_path}"')
    print("  2. Re-run the data-preparation cells in the notebook to rebuild")
    print("     data/data.joblib, inference/tokenizer.joblib, etc.")
    print("  3. Train the model.")


if __name__ == "__main__":
    main()
