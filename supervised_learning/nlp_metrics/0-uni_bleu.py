#!/usr/bin/env python3
"""
calculates the unigram BLEU score for a sentence
"""
import numpy as np


def countWords(wordList):
    """
    count the occurrences of each word in a list

    wordList is the list of words to count
    wordCounts is the dictionary of words to add the counts to

    Returns: wordCounts dictionary w/ words as keys and their counts as values
    """
    wordCounts = {}
    for word in wordList:
        wordCounts[word] = wordCounts.get(word, 0) + 1
    return wordCounts


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    unigrams: single words

    what is BLEU?
    BLEU = Bilingual Evaluation Understudy
    BLEU provides a numeric score (ranging typically between 0 and 1)
        as a measure of the quality of machine translation,
        with higher scores indicating better translations

    BLEU assesses translation quality by comparing n-grams
        (contiguous sequences of n items from a given sample of text or speech)
        in the machine-translated text to n-grams in a reference translation
    it counts the matches, adjusting for coincidental matches with
        a brevity penalty

    brevity penalty:
    the brevity penalty is applied to prevent overly short translations
        from receiving high scores
    it penalizes translations that are much shorter than
        their reference translations

    Returns: the unigram BLEU score
    """
    # count how many times each word occurs in the sentence
    wordCounts = countWords(sentence)

    # maxReferenceCounts will store the highest count
    #   of each word across all references
    # this is important for clipping later
    #   ensuring words are not over counted
    maxReferenceCounts = {word: 0 for word in wordCounts}

    # loop to update maxReferenceCounts according to each reference translation
    # compare each word in a sentence against each reference translation
    for reference in references:
        # count occurences of each word in the current reference
        referenceCounts = countWords(reference)
        # loop over words in the sentence
        for word in wordCounts:
            # if a word is more frequent in this reference than seen before,
            #   update its max count in maxReferenceCounts
            maxReferenceCounts[word] = max(
                maxReferenceCounts[word], referenceCounts.get(word, 0)
            )

    # clip each word count in the sentence by
    #   its max count found in the references
    # this prevents inflating the score by repeating common words
    clippedCounts = {
        word: min(count, maxReferenceCounts[word]) for word,
        count in wordCounts.items()
    }

    # calculate precision as the ratio of clipped counts
    #   to total counts in the sentence
    # this measures how accurately the sentence captures common
    #   unigrams in the references
    precision = sum(clippedCounts.values()) / sum(wordCounts.values())

    sentenceLength = len(sentence)
    # create a list of lengths for each reference translation
    referenceLengths = [len(reference) for reference in references]
    # find the reference length closest to the sentence length
    # this will be used to calculate the brevity penalty
    closestReferenceLength = None
    minDiff = float("inf")
    for refLength in referenceLengths:
        diff = abs(refLength - sentenceLength)
        if diff < minDiff:
            minDiff = diff
            closestReferenceLength = refLength
    # calculate the brevity penalty based on sentence length
    #   and closest reference length
    brevityPenalty = (
        # penalize if sentence is shorter than closest reference
        #   to avoid favoring short, uninformative sentences
        np.exp(1 - closestReferenceLength / sentenceLength)
        if sentenceLength < closestReferenceLength
        # else no penalty (1)
        else 1
    )

    # calculate the final BLEU score by multiplying
    #   the precision by the brevity penalty
    BLEUscore = brevityPenalty * precision
    return BLEUscore
