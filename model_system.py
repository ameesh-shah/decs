from abc import ABC, abstractmethod

class ModelSystem():

    def __init__(self, learner, verifier):
        self.learner = learner
        self.verifier = verifier
        self.current_candidate = None

    def train_candidate(self, input_dataset):
        self.current_candidate = self.learner.synthesize_candidate(input_dataset)
        return self.current_candidate

    def check_candidate(self):
        verification = self.verifier.verify(self.current_candidate)
        return verification

class Learner(ABC):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def synthesize_candidate(self, input_dataset):
        pass

class Verifier(ABC):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def verify(self, candidate):
        pass
