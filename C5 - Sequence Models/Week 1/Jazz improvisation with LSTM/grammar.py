'''
Author:     Ji-Sung Kim, Evan Chow
Project:    jazzml / (used in) deepjazz
Purpose:    Extract, manipulate, process musical grammar

Directly taken then cleaned up from Evan Chow's jazzml, 
https://github.com/evancchow/jazzml,with permission.
'''

from collections import OrderedDict, defaultdict
from itertools import groupby
from music21 import *
import copy, random, pdb

#from preprocess import *

''' Helper function to determine if a note is a scale tone. '''
def __is_scale_tone(chord, note):
    # Method: generate all scales that have the chord notes th check if note is
    # in names

    # Derive major or minor scales (minor if 'other') based on the quality
    # of the chord.
    scaleType = scale.DorianScale() # i.e. minor pentatonic
    if chord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(chord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Get note name. Return true if in the list of note names.
    noteName = note.name
    return (noteName in allNoteNames)

''' Helper function to determine if a note is an approach tone. '''
def __is_approach_tone(chord, note):
    # Method: see if note is +/- 1 a chord tone.

    for chordPitch in chord.pitches:
        stepUp = chordPitch.transpose(1)
        stepDown = chordPitch.transpose(-1)
        if (note.name == stepDown.name or 
            note.name == stepDown.getEnharmonic().name or
            note.name == stepUp.name or
            note.name == stepUp.getEnharmonic().name):
                return True
    return False

''' Helper function to determine if a note is a chord tone. '''
def __is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

''' Helper function to generate a chord tone. '''
def __generate_chord_tone(lastChord):
    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
    return note.Note(random.choice(lastChordNoteNames))

''' Helper function to generate a scale tone. '''
def __generate_scale_tone(lastChord):
    # Derive major or minor scales (minor if 'other') based on the quality
    # of the lastChord.
    scaleType = scale.WeightedHexatonicBlues() # minor pentatonic
    if lastChord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Return a note (no octave here) in a scale that matches the lastChord.
    sNoteName = random.choice(allNoteNames)
    lastChordSort = lastChord.sortAscending()
    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
    return sNote

''' Helper function to generate an approach tone. '''
def __generate_approach_tone(lastChord):
    sNote = __generate_scale_tone(lastChord)
    aNote = sNote.transpose(random.choice([1, -1]))
    return aNote

''' Helper function to generate a random tone. '''
def __generate_arbitrary_tone(lastChord):
    return __generate_scale_tone(lastChord) # fix later, make random note.


''' Given the notes in a measure ('measure') and the chords in that measure
    ('chords'), generate a list of abstract grammatical symbols to represent 
    that measure as described in GTK's "Learning Jazz Grammars" (2009). 

    Inputs: 
    1) "measure" : a stream.Voice object where each element is a
        note.Note or note.Rest object.

        >>> m1
        <music21.stream.Voice 328482572>
        >>> m1[0]
        <music21.note.Rest rest>
        >>> m1[1]
        <music21.note.Note C>

        Can have instruments and other elements, removes them here.

    2) "chords" : a stream.Voice object where each element is a chord.Chord.

        >>> c1
        <music21.stream.Voice 328497548>
        >>> c1[0]
        <music21.chord.Chord E-4 G4 C4 B-3 G#2>
        >>> c1[1]
        <music21.chord.Chord B-3 F4 D4 A3>

        Can have instruments and other elements, removes them here. 

    Outputs:
    1) "fullGrammar" : a string that holds the abstract grammar for measure.
        Format: 
        (Remember, these are DURATIONS not offsets!)
        "R,0.125" : a rest element of  (1/32) length, or 1/8 quarter note. 
        "C,0.125<M-2,m-6>" : chord note of (1/32) length, generated
                             anywhere from minor 6th down to major 2nd down.
                             (interval <a,b> is not ordered). '''

def parse_melody(fullMeasureNotes, fullMeasureChords):
    # Remove extraneous elements.x
    measure = copy.deepcopy(fullMeasureNotes)
    chords = copy.deepcopy(fullMeasureChords)
    measure.removeByNotOfClass([note.Note, note.Rest])
    chords.removeByNotOfClass([chord.Chord])

    # Information for the start of the measure.
    # 1) measureStartTime: the offset for measure's start, e.g. 476.0.
    # 2) measureStartOffset: how long from the measure start to the first element.
    measureStartTime = measure[0].offset - (measure[0].offset % 4)
    measureStartOffset  = measure[0].offset - measureStartTime

    # Iterate over the notes and rests in measure, finding the grammar for each
    # note in the measure and adding an abstract grammatical string for it. 

    fullGrammar = ""
    prevNote = None # Store previous note. Need for interval.
    numNonRests = 0 # Number of non-rest elements. Need for updating prevNote.
    for ix, nr in enumerate(measure):
        # Get the last chord. If no last chord, then (assuming chords is of length
        # >0) shift first chord in chords to the beginning of the measure.
        try: 
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]
        except IndexError:
            chords[0].offset = measureStartTime
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]

        # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.
        # Dealing with solo notes here. If unexpected chord: still call 'C'.
        elementType = ' '
        # R: First, check if it's a rest. Clearly a rest --> only one possibility.
        if isinstance(nr, note.Rest):
            elementType = 'R'
        # C: Next, check to see if note pitch is in the last chord.
        elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
            elementType = 'C'
        # L: (Complement tone) Skip this for now.
        # S: Check if it's a scale tone.
        elif __is_scale_tone(lastChord, nr):
            elementType = 'S'
        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
        elif __is_approach_tone(lastChord, nr):
            elementType = 'A'
        # X: Otherwise, it's an arbitrary tone. Generate random note.
        else:
            elementType = 'X'

        # SECOND, get the length for each element. e.g. 8th note = R8, but
        # to simplify things you'll use the direct num, e.g. R,0.125
        if (ix == (len(measure)-1)):
            # formula for a in "a - b": start of measure (e.g. 476) + 4
            diff = measureStartTime + 4.0 - nr.offset
        else:
            diff = measure[ix + 1].offset - nr.offset

        # Combine into the note info.
        noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff

        # THIRD, get the deltas (max range up, max range down) based on where
        # the previous note was, +- minor 3. Skip rests (don't affect deltas).
        intervalInfo = ""
        if isinstance(nr, note.Note):
            numNonRests += 1
            if numNonRests == 1:
                prevNote = nr
            else:
                noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                noteDistUpper = interval.add([noteDist, "m3"])
                noteDistLower = interval.subtract([noteDist, "m3"])
                intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                    noteDistLower.directedName)
                # print "Upper, lower: %s, %s" % (noteDistUpper,
                #     noteDistLower)
                # print "Upper, lower dnames: %s, %s" % (
                #     noteDistUpper.directedName,
                #     noteDistLower.directedName)
                # print "The interval: %s" % (intervalInfo)
                prevNote = nr

        # Return. Do lazy evaluation for real-time performance.
        grammarTerm = noteInfo + intervalInfo 
        fullGrammar += (grammarTerm + " ")

    return fullGrammar.rstrip()

''' Given a grammar string and chords for a measure, returns measure notes. '''
def unparse_grammar(m1_grammar, m1_chords):
    m1_elements = stream.Voice()
    currOffset = 0.0 # for recalculate last chord.
    prevElement = None
    for ix, grammarElement in enumerate(m1_grammar.split(' ')):
        terms = grammarElement.split(',')
        currOffset += float(terms[1]) # works just fine

        # Case 1: it's a rest. Just append
        if terms[0] == 'R':
            rNote = note.Rest(quarterLength = float(terms[1]))
            m1_elements.insert(currOffset, rNote)
            continue

        # Get the last chord first so you can find chord note, scale note, etc.
        try: 
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
        except IndexError:
            m1_chords[0].offset = 0.0
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

        # Case: no < > (should just be the first note) so generate from range
        # of lowest chord note to highest chord note (if not a chord note, else
        # just generate one of the actual chord notes). 

        # Case #1: if no < > to indicate next note range. Usually this lack of < >
        # is for the first note (no precedent), or for rests.
        if (len(terms) == 2): # Case 1: if no < >.
            insertNote = note.Note() # default is C

            # Case C: chord note.
            if terms[0] == 'C':
                insertNote = __generate_chord_tone(lastChord)

            # Case S: scale note.
            elif terms[0] == 'S':
                insertNote = __generate_scale_tone(lastChord)

            # Case A: approach note.
            # Handle both A and X notes here for now.
            else:
                insertNote = __generate_approach_tone(lastChord)

            # Update the stream of generated notes
            insertNote.quarterLength = float(terms[1])
            if insertNote.octave < 4:
                insertNote.octave = 4
            m1_elements.insert(currOffset, insertNote)
            prevElement = insertNote

        # Case #2: if < > for the increment. Usually for notes after the first one.
        else:
            # Get lower, upper intervals and notes.
            interval1 = interval.Interval(terms[2].replace("<",''))
            interval2 = interval.Interval(terms[3].replace(">",''))
            if interval1.cents > interval2.cents:
                upperInterval, lowerInterval = interval1, interval2
            else:
                upperInterval, lowerInterval = interval2, interval1
            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
            numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)

            # Case C: chord note, must be within increment (terms[2]).
            # First, transpose note with lowerInterval to get note that is
            # the lower bound. Then iterate over, and find valid notes. Then
            # choose randomly from those.
            
            if terms[0] == 'C':
                relevantChordTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case S: scale note, must be within increment.
            elif terms[0] == 'S':
                relevantScaleTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_scale_tone(lastChord, currNote):
                        relevantScaleTones.append(currNote)
                if len(relevantScaleTones) > 1:
                    insertNote = random.choice([i for i in relevantScaleTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantScaleTones) == 1:
                    insertNote = relevantScaleTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case A: approach tone, must be within increment.
            # For now: handle both A and X cases.
            else:
                relevantApproachTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_approach_tone(lastChord, currNote):
                        relevantApproachTones.append(currNote)
                if len(relevantApproachTones) > 1:
                    insertNote = random.choice([i for i in relevantApproachTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantApproachTones) == 1:
                    insertNote = relevantApproachTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # update the previous element.
            prevElement = insertNote

    return m1_elements    