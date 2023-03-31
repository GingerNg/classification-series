from nltk.stem import WordNetLemmatizer

class EnProcessor(object):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, text):
        return self.lemmatizer.lemmatize(text)


class CnProcessor(object):
    def __init__(self):
        super().__init__()

