'''
Author:     Ji-Sung Kim, Evan Chow
Project:    deepjazz
Purpose:    Provide pruning and cleanup functions.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.
'''
from itertools import zip_longest
import random

from music21 import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to down num to the nearest multiple of mult. '''
def __roundDown(num, mult):
    return (float(num) - (float(num) % mult))

''' Helper function to round up num to nearest multiple of mult. '''
def __roundUp(num, mult):
    return __roundDown(num, mult) + mult

''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 
    down or up respectively to nearest multiple of mult. '''
def __roundUpDown(num, mult, upDown):
    if upDown < 0:
        return __roundDown(num, mult)
    else:
        return __roundUp(num, mult)

''' Helper function, from recipes, to iterate over list in chunks of n 
    length. '''
def __grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Smooth the measure, ensuring that everything is in standard note lengths 
    (e.g., 0.125, 0.250, 0.333 ... ). '''
def prune_grammar(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')

    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, 
            random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)

    return pruned_grammar

''' Remove repeated notes, and notes that are too close together. '''
def prune_notes(curr_notes):
    for n1, n2 in __grouper(curr_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                curr_notes.remove(n2)

    return curr_notes

''' Perform quality assurance on notes '''
def clean_up_notes(curr_notes):
    removeIxs = []
    for ix, m in enumerate(curr_notes):
        # QA1: ensure nothing is of 0 quarter note len, if so changes its len
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA2: ensure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(curr_notes) - 1)):
            if (m.offset == curr_notes[ix + 1].offset and
                isinstance(curr_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]

    return curr_notes