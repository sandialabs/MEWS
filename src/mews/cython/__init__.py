"""
The Multi-scenario Extreme Weather Simulator (MEWS)
weather module reads in, alters, and writes out weather files
to account for extreme events or climate change.
"""
from mews.cython.markov import markov_chain
from mews.cython.markov_time_dependent import markov_chain_time_dependent
