Annotator columns
- annID: unique annotatorID
- Attitude columns:
    - altruism
    - empathy
    - freeSpeech
    - harmHateSpeech
    - traditionalism
    - lingPurism
    - racism
- Demographic columns:
    - annotatorGender
    - annotatorRace: note, some of the annotators listed multiple in the small scale study
    - annotatorPolitics: -1 to 1, with -1 being left and 1 being right

Tweet- or post-level columns:
- postId
- postCategory
- Toxicity rating columns:
    - intent
    - racist
    - toany
    - toyou
    - off_avg


Only in large scale:
- annotatorMinority: free-text other minority status (only large scale study)
- annotatorAge
- targetsBlackPeople, vulgar, isAAE: booleans for whether a post belongs to a certain category or not.
- dialectColumns:
    - hispanic
    - other
    - white
    - aae
    - dialAm: which of the previous four had the highest score
- dontUnderstand: whether or not the annotator understood the post
- vulgarity columns: booleans of whether it contains words of that category
    - oi
    - oni
    - noi
- info from data source:
    - ogId
    - ogLabel
    - ogLabelToxic
    - source
- tweet: actual tweet text


Only in small scale study:
- BLMsupport
- bornUS
- divSocialCircle
- manVsWoman
- nLanguagesSpoken
- politicallyEngaged
- sexualOrientation
- socialMediaUsage
- straightOrNot
- whiteVsBlack
