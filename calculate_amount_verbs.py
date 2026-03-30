import pandas as pd
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tokenizer"])

# Paths to your CSV files
files = {
    "BNC": "./evaluation/clm_semantic/data/verb/bnc_bin.csv",
    "CANDOR": "./evaluation/clm_semantic/data/verb/candor_bin.csv",
    "WIKIPEDIA": "./evaluation/clm_semantic/data/verb/wiki_bin.csv",
    "CDS": "./evaluation/clm_semantic/data/verb/cds_bin.csv"
}

def extract_unique_lemmas(csv_path):
    df = pd.read_csv(csv_path)
    verb_lemmas = set()
    
    for s1, s2 in zip(df['sentence1'], df['sentence2']):
        tokens1 = s1.split()
        tokens2 = s2.split()
        
        # Find the differing token
        for t1, t2 in zip(tokens1, tokens2):
            if t1 != t2:
                # Lemmatize t1 using SpaCy
                doc = nlp(t1)
                lemma = doc[0].lemma_
                verb_lemmas.add(lemma)
                break  # Only one token should differ
    
    return verb_lemmas

# Calculate and print results for each corpus
for corpus, path in files.items():
    unique_lemmas = extract_unique_lemmas(path)
    print(f"{corpus}: {len(unique_lemmas)} unique verb lemmas")
    print(unique_lemmas)
    