#!/usr/bin/env python3
"""
calculates the n-gram BLEU score for a sentence

an n-gram is a contiguous sequence of n words
"""
import numpy as np


def generateNgrams(words, n):
    """
    generate n-grams from a list of words

    uses the zip function along with list slicing to create n-grams
    each n-gram is a tuple of 'n' consecutive words
        from the list, where 'n' is defined by the parameter

    creates n-grams by pairing consecutive words using the zip function
        ...zip iterates over multiple list-like objects simultaneously
    here, zip is applied to sliced versions of the original word list
    each slice starts from a different position in the list,
        offset by the n-gram size, creating overlapping segments of words
    in a trigram (n=3) scenario:
        zip combines word[0], word[1], and word[2] from the first iteration,
            then word[1], word[2], and word[3] in the next, &so on
    this effectively groups every set of 'n' consecutive words into a tuple,
        forming an n-gram

    words: list of words to generate n-grams from
    n: The size of the ngram

    returns: list of n-grams
    """
    # zipping slices of words into ngrams from the list of words
    ngrams = zip(*[words[i:] for i in range(n)])
    return list(ngrams)


def countNgrams(ngrams):
    """
    count the n-grams in a list

    tallies the occurrences of each unique n-gram in the given list

    ngrams: list of n-grams to count

    returns: dictionary with n-grams as keys and their counts as values
    """
    ngramCounts = {}
    # tally up the ngrams
    for ngram in ngrams:
        ngramCounts[ngram] = ngramCounts.get(ngram, 0) + 1
    return ngramCounts


def calculateClippedCounts(sentenceNgramCounts, references, n):
    """
    takes the ngram counts of a sentence and clips them
        based on their occurrences in reference translations
    first, it finds the maximum count of each ngram across all references,
        then it clips the sentence's ngram counts to these maxima
    this is essential in BLEU score calculation to prevent over counting
        ngrams present in multiple references

    sentenceNgramCounts: dict with counts of ngrams in the sentence
    references: list of reference translations
    n: size of ngrams

    Returns: clipped ngram counts of the sentence
    """
    maxRefNgramCounts = {}
    # loop through each reference translation
    for reference in references:
        # first, we make ngrams out of the current reference
        referenceNgrams = generateNgrams(reference, n)
        # next, we count how many times each ngram shows up in this reference
        referenceNgramCounts = countNgrams(referenceNgrams)
        # for each ngram, we want to keep the maximum count from any reference
        # loop through each ngram we found in our sentence
        for ngram in sentenceNgramCounts:
            # did we see this ngram in the current reference?
            # if yes, let's see if it's the new high score
            #   for this ngram across all references
            maxRefNgramCounts[ngram] = max(
                maxRefNgramCounts.get(ngram, 0),
                referenceNgramCounts.get(ngram, 0)
            )

    # all references are checked, so clip the counts in the sentence
    # can't score more than the highest score an ngram got in the references
    clippedCounts = {
        ngram: min(count, maxRefNgramCounts.get(ngram, 0))
        for ngram, count in sentenceNgramCounts.items()
    }
    return clippedCounts


def findClosestLength(sentenceLength, referenceLengths):
    """
    find the closest reference length to the length of a given sentence

    compares length of a sentence with a list of lengths
        from reference translations
    finds and returns the length of the reference
        that is closest to the sentence length
    this is useful in calculating brevity penalty in BLEU score calculation,
        where matching the lengths closely impacts the penalty factor

    sentenceLength: length of the sentence being evaluated
    referenceLengths: list of lengths of reference translations

    Returns: length of the closest reference translation
    """
    # find the best-fit length from our references for the sentence
    closestLength = referenceLengths[
        0
    ]  # starting with the first reference length as our initial guess
    # loop through the reference lengths
    for referenceLength in referenceLengths:
        # is this reference length closer to our sentence length
        #   than our current best guess?
        if abs(referenceLength - sentenceLength) \
                < abs(closestLength - sentenceLength):
            closestLength = referenceLength
    return closestLength


def calculateBrevityPenalty(sentenceLength, referenceLengths):
    """
    calculate the brevity penalty for BLEU score calculation

    this is an important part of the BLEU score calculation,
        which penalizes shorter translated sentences to ensure
        closeness in length to the reference translations
    the penalty is based on the ratio of the closest
        reference length to the sentence length

    sentenceLength: length of the sentence being evaluated
    referenceLengths: list of lengths of reference translations

    Returns: calculated brevity penalty
    """
    closestReferenceLength = findClosestLength(
        sentenceLength, referenceLengths
        )
    # the penalty math: if the sentence is shorter than the reference,
    #   apply a penalty
    brevityPenalty = (
        np.exp(1 - closestReferenceLength / sentenceLength)
        if sentenceLength < closestReferenceLength
        else 1  # no penalty if sentence is longer or as long as closest ref
    )
    return brevityPenalty


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence

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

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    # break sentence into ngrams
    sentenceNgrams = generateNgrams(sentence, n)
    # count how many times each ngram appears
    sentenceNgramCounts = countNgrams(sentenceNgrams)

    # no n-grams (such as an empty sentence), returns score of 0
    if not sentenceNgramCounts:
        return 0

    # clip each word count in the sentence by
    #   its max count found in the references
    # this prevents inflating the score by repeating common words
    clippedCounts = calculateClippedCounts(sentenceNgramCounts, references, n)
    # calculate precision as the ratio of clipped counts
    #   to total counts in the sentence
    # this measures how accurately the sentence captures common
    #   unigrams in the references
    precision = sum(clippedCounts.values()) / sum(sentenceNgramCounts.values())

    # gather lengths of all our reference translations
    referenceLengths = [len(ref) for ref in references]
    # calculate brevity penalty based on sentence length and reference lengths
    brevityPenalty = calculateBrevityPenalty(len(sentence), referenceLengths)

    # finally, calculate BLEU score!
    BLEUscore = brevityPenalty * precision

    # return sentence's ngram BLEU score
    return BLEUscore
