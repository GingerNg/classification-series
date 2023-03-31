from typing import List


class Pipeline(object):
    def __init__(self, steps: List):
        super().__init__()
        self.steps = steps

    def transform(self):
        pass
