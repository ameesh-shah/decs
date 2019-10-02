
class GroundTruth():

    def __init__(self, is_queryable, datapairs, negative_data=[]):
        self.is_queryable = is_queryable
        self.positive_dataset = datapairs
        self.negative_dataset = negative_data

    def get_dataset(self):
        return self.dataset

    def is_queryable(self):
        return self.is_queryable

    def query(self):
        if self.is_queryable():
            print("Not implemented: get agent")
        else:
            print("ERROR: ground truth is not queryable")
