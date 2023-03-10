From https://maartensap.com/racial-bias-hatespeech/
  - AnnWithAttitues
  - davidson_dial.csv
  - founta_all_dial.csv
  - sap2019risk_mTurkExperiment.csv

train_data.csv:
  - generated from Davidson + Founta
  - only use the possibly biased original labels from these datasets
  - excluded tweets in test data
  - Total # tweets: 114774
  - columns: 
    - tweet: text, 
    - davidson_id:  
    - founta_id: 
    - og_label: original label from davidson and founta
    - label: davidson - ['neither_rel'=0 'hate_speech_rel'=1 'offensive_language_rel'=1 nan]; founta - [nan 'normal'=0 'spam'=0 'hateful'=1 'abusive'=1]

test_data_relaxed.csv:
  - consider all the dialect/race primed answers to be unbiased ground truth labels

test_data_strict.csv:
  - consider only the answers where annotator's race is the same as the author's race
  - first get dialect/race primed answers, then propagate their conditions to the same tweets with "text-only" conditions (they are used as controls in the experiments in the original paper)
  - then select tweets where annotator's race is the same as the author's race
  - then aggregate to get majority vote