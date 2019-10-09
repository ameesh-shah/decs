
class GroundTruth():

    def __init__(self, is_queryable, positive_examples, negative_examples = []):
        self.is_queryable = is_queryable
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples

    def get_dataset(self):
        return self.positive_examples, self.negative_examples

    def is_queryable(self):
        return self.is_queryable

    def query(self, input):
        if self.is_queryable():
            print("Not implemented: get agent")
        else:
            print("ERROR: ground truth is not queryable")
