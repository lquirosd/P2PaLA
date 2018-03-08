from __future__ import print_function
from __future__ import division
from builtins import range

import numpy as np
from page_xml.xmlPAGE import pageData

#--- GOAL Oriented Perfomance Evaluation Methodology for page Segmentation
#--- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7333768

#--- ICDAR2009 Page Segmentation Competition Evaluation metrics
#--- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5277763
#--- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4377117

#--- Pixel Base P/R ICDAR2007

#--- Lavenshtein edit distance
def levenshtein(source, target):
    """
    levenshtein edit distance using
    addcost=delcost=subcost=1
    Borrowed form: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

