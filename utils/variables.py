import os
import spacy

# Load SpaCy model
nlp_eng = spacy.load('en_core_web_sm')
'''nlp_fr = spacy.load('fr_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')'''

# Define the root directory for your project
BASE_DIR = os.path.expanduser('~/Desktop/childes_vs_wiki')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation', 'test_suites')
EVAL_DIR_SCRIPTS = os.path.join(BASE_DIR, 'evaluation','scripts')
PARSED_DATASETS = os.path.join(BASE_DIR, 'parsed_datasets')
BARPLOTS_DIR = os.path.join(EVAL_DIR_SCRIPTS, 'barplots')

# Define subdirectories and files relative to the base directory
TRAINED_MODELS = os.path.join('./Desktop/babyLM_multilingual', 'models_trained_recent', 'lastly_trained')
CORPORA_DIR = os.path.join(BASE_DIR, 'corpora') 
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


CLAMS_DIR = os.path.join(EVAL_DIR, 'clams')
CLAMS_FOLDER_ENG = os.path.join(CLAMS_DIR, 'en_evalset_ok')
CLAMS_FOLDER_FR = os.path.join(CLAMS_DIR,'fr_evalset_ok')
CLAMS_FOLDER_DE = os.path.join(CLAMS_DIR, 'de_evalset_ok')

CLAMS_FILTERED_FOLDER_ENG = os.path.join(CLAMS_DIR, 'en_evalset_filtered')
CLAMS_FILTERED_FOLDER_FR = os.path.join(CLAMS_DIR, 'fr_evalset_filtered')
CLAMS_FILTERED_FOLDER_DE = os.path.join(CLAMS_DIR, 'de_evalset_filtered')
CLAMS_EXCLUDED_FOLDER_ENG = os.path.join(CLAMS_DIR, 'en_evalset_excluded')
CLAMS_EXCLUDED_FOLDER_FR = os.path.join(CLAMS_DIR, 'fr_evalset_excluded')
CLAMS_EXCLUDED_FOLDER_DE = os.path.join(CLAMS_DIR, 'de_evalset_excluded')


## original dataset
ZORRO_DIR = os.path.join(EVAL_DIR, 'zorro')
BLIMP_DIR = os.path.join(EVAL_DIR, 'blimp')

## filtered dataset
ZORRO_FILTERED_FOLDER = os.path.join(EVAL_DIR, 'zorro_filtered')
BLIMP_FILTERED_FOLDER = os.path.join(EVAL_DIR, 'blimp_filtered')

ZORRO_EXCLUDED_FOLDER = os.path.join(EVAL_DIR, 'zorro_excluded')
BLIMP_EXCLUDED_FOLDER = os.path.join(EVAL_DIR, 'blimp_excluded')

## fit clams
FITCLAMS_DIR = os.path.join(EVAL_DIR, 'fit_clams')
FITCLAMS_ENG_childes = os.path.join(FITCLAMS_DIR, 'en_', 'childes')
FITCLAMS_FR_childes = os.path.join(FITCLAMS_DIR, 'fr_', 'childes')
FITCLAMS_DE_childes = os.path.join(FITCLAMS_DIR, 'de_', 'childes')
FITCLAMS_ENG_wikipedia = os.path.join(FITCLAMS_DIR, 'en_', 'wiki')
FITCLAMS_FR_wikipedia = os.path.join(FITCLAMS_DIR, 'fr_', 'wiki')
FITCLAMS_DE_wikipedia = os.path.join(FITCLAMS_DIR, 'de_', 'wiki')


## ao-childes files in the three languages
AO_CHILDES_ENGLISH= os.path.join(CORPORA_DIR, 'english', 'aochildes','aoenglish_df.csv')
AO_CHILDES_FRENCH = os.path.join(CORPORA_DIR, 'french', 'aochildes','aofrench_df.csv')
AO_CHILDES_GERMAN = os.path.join(CORPORA_DIR, 'german', 'aochildes','aogerman_df.csv')


## wikipedia files 
WIKIPEDIA_ENG = os.path.join(CORPORA_DIR, 'english', 'wikipedia', 'wikipedia_final.csv')
WIKIPEDIA_FR = os.path.join(CORPORA_DIR, 'french', 'wikipedia', 'wikipedia_final.csv')
WIKIPEDIA_DE = os.path.join(CORPORA_DIR, 'german', 'wikipedia', 'wikipedia_final.csv')

TRAINING_CHILDES_ENG = os.path.join(DATASET_DIR, 'english', 'random', 'train.csv')
TRAINING_CHILDES_FR = os.path.join(DATASET_DIR, 'french', 'random', 'train.csv')
TRAINING_CHILDES_DE = os.path.join(DATASET_DIR, 'german', 'random', 'train.csv')
TRAINING_WIKI_ENG = os.path.join(DATASET_DIR, 'english', 'wikipedia', 'train.csv')
TRAINING_WIKI_FR = os.path.join(DATASET_DIR, 'french', 'wikipedia', 'train.csv')
TRAINING_WIKI_DE = os.path.join(DATASET_DIR, 'german', 'wikipedia', 'train.csv')

VALIDATION_CHILDES_ENG = os.path.join(DATASET_DIR, 'english', 'random', 'validation_in_context.csv')
VALIDATION_CHILDES_FR = os.path.join(DATASET_DIR, 'french', 'random', 'validation_in_context.csv')
VALIDATION_CHILDES_DE = os.path.join(DATASET_DIR, 'german', 'random', 'validation_in_context.csv')
VALIDATION_WIKI_ENG = os.path.join(DATASET_DIR, 'english', 'wikipedia', 'validation.csv')
VALIDATION_WIKI_FR = os.path.join(DATASET_DIR, 'french', 'wikipedia', 'validation.csv')
VALIDATION_WIKI_DE = os.path.join(DATASET_DIR, 'german', 'wikipedia', 'validation.csv')

FIT_CLAMS_GENERATION = os.path.join(BASE_DIR, 'fitclams_generation')
INTERSECTED_VOCAB_EN = os.path.join(FIT_CLAMS_GENERATION,"intersected_vocabulary","intersected_vocabulary_eng.txt")
INTERSECTED_VOCAB_FR = os.path.join(FIT_CLAMS_GENERATION, "intersected_vocabulary","intersected_vocabulary_fr.txt")
INTERSECTED_VOCAB_DE = os.path.join(FIT_CLAMS_GENERATION, "intersected_vocabulary","intersected_vocabulary_de.txt")


JSON_RESULT_ZORRO_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm','json_zorro_ok')
JSON_RESULT_BLIMP_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm','json_blimp_ok')
JSON_RESULT_clams_ENG_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_en_clams_ok')
JSON_RESULT_clams_DE_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_de_clams_ok')
JSON_RESULT_clams_FR_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_fr_clams_ok')

JSON_RESULT_ZORRO_mlm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_mlm','json_zorro_ok')
JSON_RESULT_BLIMP_mlm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_mlm','json_blimp_ok')
JSON_RESULT_clams_ENG_mlm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_mlm', 'json_en_clams_ok')
JSON_RESULT_clams_DE_mlm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_mlm', 'json_de_clams_ok')
JSON_RESULT_clams_FR_mlm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_mlm', 'json_fr_clams_ok')


JSON_RESULT_fitclams_ENG_childes_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_en_newclams_childes')
JSON_RESULT_fitclams_DE_childes_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_de_newclams_childes')
JSON_RESULT_fitclams_FR_childes_clm  = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_fr_newclams_childes')
JSON_RESULT_fitclams_ENG_wiki_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_en_newclams_wiki')
JSON_RESULT_fitclams_DE_wiki_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_de_newclams_wiki')
JSON_RESULT_fitclams_FR_wiki_clm = os.path.join(EVAL_DIR_SCRIPTS, 'json_results_clm', 'json_fr_newclams_wiki')



PLOT_RESULTS_FOLDER_CLM = os.path.join(EVAL_DIR_SCRIPTS, 'plot_clm_seeds')
PLOT_RESULTS_FOLDER_MLM = os.path.join(EVAL_DIR_SCRIPTS, 'plot_mlm_seeds')


# ENGLISH
UNIGRAM_CHILDES_ENG = os.path.join(BASE_DIR, "n_grams_frequency/eng/childes/unigram/unigram_freq_childes.csv")
UNIGRAM_WIKI_ENG = os.path.join(BASE_DIR, "n_grams_frequency/eng/wikipedia/unigram/unigram_freq_wikipedia.csv")
UNIGRAM_DEP_CHILDES_ENG = os.path.join(BASE_DIR, "n_grams_frequency/eng/parsed_frequency_childes/unigram_deprel_freq.csv")
UNIGRAM_DEP_WIKI_ENG = os.path.join(BASE_DIR, "n_grams_frequency/eng/parsed_frequency_wiki/unigram_deprel_freq.csv")

# FRENCH
UNIGRAM_CHILDES_FR = os.path.join(BASE_DIR, "n_grams_frequency/fr/childes/unigram/unigram_freq_childes.csv")
UNIGRAM_WIKI_FR = os.path.join(BASE_DIR, "n_grams_frequency/fr/wikipedia/unigram/unigram_freq_wikipedia.csv")
UNIGRAM_DEP_CHILDES_FR = os.path.join(BASE_DIR, "n_grams_frequency/fr/parsed_frequency_childes/unigram_deprel_freq.csv")
UNIGRAM_DEP_WIKI_FR = os.path.join(BASE_DIR, "n_grams_frequency/fr/parsed_frequency_wiki/unigram_deprel_freq.csv")

# GERMAN
UNIGRAM_CHILDES_DE = os.path.join(BASE_DIR, "n_grams_frequency/de/childes/unigram/unigram_freq_childes.csv")
UNIGRAM_WIKI_DE = os.path.join(BASE_DIR, "n_grams_frequency/de/wikipedia/unigram/unigram_freq_wikipedia.csv")
UNIGRAM_DEP_CHILDES_DE = os.path.join(BASE_DIR, "n_grams_frequency/de/parsed_frequency_childes/unigram_deprel_freq.csv")
UNIGRAM_DEP_WIKI_DE = os.path.join(BASE_DIR, "n_grams_frequency/de/parsed_frequency_wiki/unigram_deprel_freq.csv")


# Define file paths
MISSING_WORDS_ZORRO = os.path.join(EVAL_DIR,'missing_words_eng_zorro_finale.json')
MISSING_WORDS_BLIMP = os.path.join(EVAL_DIR, 'missing_words_eng_blimp_finale.json')
MISSING_WORDS_CLAMS_ENG = os.path.join(EVAL_DIR,'missing_words_eng_clams.json')
MISSING_WORDS_CLAMS_FR = os.path.join(EVAL_DIR,'missing_words_fr_clams.json')
MISSING_WORDS_CLAMS_DE = os.path.join(EVAL_DIR,'missing_words_de_clams.json')
EVAL_PROB_DIRS = os.path.join(BASE_DIR, 'evaluation_probabilities')

SCORE_DIRS = {
    "clams": os.path.join(EVAL_PROB_DIRS, "CLAMS_SCORES"),
    "blimp": os.path.join(EVAL_PROB_DIRS, "BLIMP_SCORES"),
    "zorro": os.path.join(EVAL_PROB_DIRS, "ZORRO_SCORES"),
}

SCORE_DIR_FITCLAMS = os.path.join(EVAL_PROB_DIRS, "FIT_CLAMS")
VALIDATION_LOSS_PLOTS = os.path.join(EVAL_DIR_SCRIPTS, 'validation_loss_plots')
BABYBERTA_DIR = os.path.join(BASE_DIR, 'BabyBERTa')

EXTRACTED_WORDS = os.path.join(BASE_DIR, "fitclams_generation/extracted_words")

desired_order = [
    "Agrmt in\nlong VP coords",
    "Agrmt in\nobj rel clauses 1",
     "Agrmt in\nobj rel clauses 2",
     "Agrmt in\nprep phrases",
    "Simple\nAgrmt",
    "Agrmt in\nsubj rel clauses",
    "Agrmt in\nVP coords"]

paradigm_mapping = {
            "long_vp_coord": "Agrmt in\nlong VP coords",
            "simple_agrmt": "Simple\nAgrmt",
            "subj_rel": "Agrmt in\nsubj rel clauses",
            "obj_rel_across_anim": "Agrmt across\nobj rel clauses",
            "obj_rel_within_anim": "Agrmt within\nobj rel clauses",
            "prep_anim": "Agrmt in\nprep phrases",
            "vp_coord": "Agrmt in\nVP coords"
        }

paradigms_name = {
    "long_vp_coord.txt": "Agrmt in long VP coords",
    "simple_agrmt.txt": "Simple Agrmt",
    "subj_rel.txt": "Agrmt in subj rel clauses",
    "obj_rel_across_anim.txt": "Agrmt in obj rel clauses 1",
    "obj_rel_within_anim.txt": "Agrmt in obj rel clauses 2",
    "prep_anim.txt": "Agrmt in prep phrases",
    "vp_coord.txt": "Agrmt in VP coordinates",
    'agreement_subject_verb-across_relative_clause.txt': 'Agreement: Subject-Verb Across Relative Clause',
    'ellipsis-n_bar.txt': 'Ellipsis: N-Bar Structures',
    'island-effects-coordinate_structure_constraint.txt': 'Island Effects: Coordinate Structure Constraint',
    'agreement_subject_verb-across_prepositional_phrase.txt': 'Agreement: Subject-Verb Across Prepositional Phrase',
    'filler-gap-wh_question_subject.txt': 'Filler-Gap Dependency: Wh-Question (Subject)',
    'case-subjective_pronoun.txt': 'Case: Subjective Pronoun',
    'agreement_subject_verb-in_simple_question.txt': 'Agreement: Subject-Verb in Simple Question',
    'quantifiers-superlative.txt': 'Quantifiers: Superlatives',
    'local_attractor-in_question_with_aux.txt': 'Agreement Attractor: In Question with Auxiliary',
    'filler-gap-wh_question_object.txt': 'Filler-Gap Dependency: Wh-Question (Object)',
    'npi_licensing-only_npi_licensor.txt': 'NPI Licensing: "Only" as Licensor',
    'argument_structure-transitive.txt': 'Argument Structure: Transitive Verbs',
    'anaphor_agreement-pronoun_gender.txt': 'Anaphor Agreement: Pronoun Gender',
    'npi_licensing-matrix_question.txt': 'NPI Licensing: Matrix Question',
    'island-effects-adjunct_island.txt': 'Island Effects: Adjunct Island',
    'argument_structure-swapped_arguments.txt': 'Argument Structure: Swapped Arguments',
    'agreement_subject_verb-in_question_with_aux.txt': 'Agreement: Subject-Verb in Question with Auxiliary',
    'agreement_determiner_noun-across_1_adjective.txt': 'Agreement: Determiner-Noun Across One Adjective',
    'binding-principle_a.txt': 'Binding: Principle A',
    'quantifiers-existential_there.txt': 'Quantifiers: Existential "There"',
    'argument_structure-dropped_argument.txt': 'Argument Structure: Dropped Argument',
    'irregular-verb.txt': 'Morphology: Irregular Verb Forms',
    'agreement_determiner_noun-between_neighbors.txt': 'Agreement: Determiner-Noun Between Neighbors','inchoative.txt': 'Inchoative Verbs',
    'complex_NP_island.txt': 'Complex NP Island',
    'coordinate_structure_constraint_object_extraction.txt': 'Coord. Structure Constraint (Object Extraction)',
    'irregular_past_participle_verbs.txt': 'Irregular Past Participles (Verbs)',
    'left_branch_island_echo_question.txt': 'Left Branch Island (Echo Question)',
    'determiner_noun_agreement_with_adj_2.txt': 'Det-N Agreement w/ Adjective 2',
    'determiner_noun_agreement_with_adjective_1.txt': 'Det-N Agreement w/ Adjective 1',
    'principle_A_c_command.txt': 'Principle A (C-command)',
    'wh_vs_that_no_gap.txt': 'Wh vs That (No Gap)',
    'left_branch_island_simple_question.txt': 'Left Branch Island (Simple Question)',
    'distractor_agreement_relative_clause.txt': 'Attractor Agreement in RC',
    'animate_subject_trans.txt': 'Animacy: Transitive Subject',
    'determiner_noun_agreement_with_adj_irregular_1.txt': 'Det-N Agreement w/ Irreg. Adj. 1',
    'tough_vs_raising_1.txt': 'Tough vs Raising 1',
    'animate_subject_passive.txt': 'Animacy: Passive Subject',
    'wh_vs_that_with_gap_long_distance.txt': 'Wh vs That (Gap, Long-Distance)',
    'drop_argument.txt': 'Argument Dropping',
    'determiner_noun_agreement_with_adj_irregular_2.txt': 'Det-N Agreement w/ Irreg. Adj. 2',
    'tough_vs_raising_2.txt': 'Tough vs Raising 2',
    'only_npi_scope.txt': 'NPI Scope: Only',
    'transitive.txt': 'Transitive Verbs',
    'wh_questions_object_gap.txt': 'Wh-Questions (Object Gap)',
    'anaphor_gender_agreement.txt': 'Anaphor Gender Agreement',
    'wh_vs_that_with_gap.txt': 'Wh vs That (With Gap)',
    'irregular_past_participle_adjectives.txt': 'Irregular Past Participles (Adj.)',
    'sentential_negation_npi_licensor_present.txt': 'Sentential Negation: NPI Licensor',
    'principle_A_case_2.txt': 'Principle A (Case 2)',
    'passive_1.txt': 'Passive 1',
    'wh_vs_that_no_gap_long_distance.txt': 'Wh vs That (No Gap, Long-Distance)',
    'principle_A_case_1.txt': 'Principle A (Case 1)',
    'passive_2.txt': 'Passive 2',
    'existential_there_subject_raising.txt': 'Existential There (Subject Raising)',
    'existential_there_quantifiers_1.txt': 'Existential There: Quantifiers 1',
    'irregular_plural_subject_verb_agreement_1.txt': 'Irreg. Plural SV Agreement 1',
    'wh_questions_subject_gap_long_distance.txt': 'Wh-Questions (Subj. Gap, Long-Distance)',
    'wh_questions_subject_gap.txt': 'Wh-Questions (Subject Gap)',
    'existential_there_quantifiers_2.txt': 'Existential There: Quantifiers 2',
    'irregular_plural_subject_verb_agreement_2.txt': 'Irreg. Plural SV Agreement 2',
    'existential_there_object_raising.txt': 'Existential There (Object Raising)',
    'sentential_negation_npi_scope.txt': 'Sentential Negation: NPI Scope',
    'wh_island.txt': 'Wh-Island',
    'superlative_quantifiers_1.txt': 'Superlative Quantifiers 1',
    'distractor_agreement_relational_noun.txt': 'Attractor Agreement: Relational Noun',
    'ellipsis_n_bar_1.txt': 'Ellipsis: N-Bar 1',
    'adjunct_island.txt': 'Adjunct Island',
    'principle_A_reconstruction.txt': 'Principle A (Reconstruction)',
    'superlative_quantifiers_2.txt': 'Superlative Quantifiers 2',
    'ellipsis_n_bar_2.txt': 'Ellipsis: N-Bar 2',
    'causative.txt': 'Causatives',
    'npi_present_1.txt': 'NPI Licensing 1',
    'regular_plural_subject_verb_agreement_2.txt': 'Regular Plural SV Agreement 2',
    'determiner_noun_agreement_2.txt': 'Det-N Agreement 2',
    'npi_present_2.txt': 'NPI Licensing 2',
    'regular_plural_subject_verb_agreement_1.txt': 'Regular Plural SV Agreement 1',
    'intransitive.txt': 'Intransitive Verbs',
    'anaphor_number_agreement.txt': 'Anaphor Number Agreement',
    'determiner_noun_agreement_1.txt': 'Det-N Agreement 1',
    'matrix_question_npi_licensor_present.txt': 'Matrix Question: NPI Licensor',
    'expletive_it_object_raising.txt': 'Expletive It (Object Raising)',
    'determiner_noun_agreement_irregular_1.txt': 'Det-N Agreement Irregular 1',
    'principle_A_domain_1.txt': 'Principle A (Domain 1)',
    'coordinate_structure_constraint_complex_left_branch.txt': 'Coord. Struct. Constraint (Complex Left Branch)',
    'sentential_subject_island.txt': 'Sentential Subject Island',
    'only_npi_licensor_present.txt': 'Only as NPI Licensor',
    'principle_A_domain_3.txt': 'Principle A (Domain 3)',
    'principle_A_domain_2.txt': 'Principle A (Domain 2)',
    'determiner_noun_agreement_irregular_2.txt': 'Det-N Agreement Irregular 2'}