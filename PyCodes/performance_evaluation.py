from collections import OrderedDict
from backend_configuration import *

class VisualPerformanceEvaluation(VisualConfiguration):

    def __init__(self):
        super().__init__()
        self._runs = {}
        self._evaluations_by_nmissing = {}
        self._evaluations_by_difficulty = {}

    @property
    def runs(self):
        return self._runs

    @property
    def evaluations_by_nmissing(self):
        return self._evaluations_by_nmissing

    @property
    def evaluations_by_difficulty(self):
        return self._evaluations_by_difficulty

class PerformanceEvaluation(VisualPerformanceEvaluation):

    def __init__(self):
        super().__init__()











##
