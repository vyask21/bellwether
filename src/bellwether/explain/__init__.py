"""The explanation layer: what caused a forecast interval breach.

The numeric core detects breaches. This half says why, and it is bound by the project's
standing constraint that the language model never produces a number. Everything quantified
here is computed in Python from stored data; the model's only job is to narrate what these
modules found and to cite it.
"""
