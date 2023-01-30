class ModelInput:
    """ WIP - Subject to change """
    def __init__(self):
        self.word_1 = []
        self.word_2 = []
        self.word_3 = []
        self.images = []

    def all_words(self):
        return [self.word_1, self.word_2, self.word_3]


class DataObject:
    def __init__(self, line_number, word, context, images):
        """
        Attributes:
         `line_number`
         `word`
         `context`
         `embeddings`
         `images`
         `goldimg`
        """
        self.line_number = line_number
        self.embeddings  = None
        # Init. attributes
        self.word, self.context, self.images, self.goldimg = \
        None, None, None, None
        # 1 word
        self.word = word
        if len(self.word) != 1:
            raise ValueError('DataObject.word length != 1')
        # 2 words
        self.context = context
        if len(self.context) < 2:
            raise ValueError(f'DataObject.context length < 2\n'
                             f'{self.__repr__()}\n')
        # 10 images
        self.images  = images
        if len(self.images) != 10:
            raise ValueError('DataObject.context length != 10')
        # 1 image
        self.goldimg = None
        self.words = None
        # Mark EOF
        pass

    def update_words(self):
        self.words = [self.word[0], self.context[0], self.context[1]]

    def create_embeddings(self, func):
        self.embeddings = func(self.word, self.context)

    def __str__(self):
        self.update_words()
        return f"\n/ Line #No: {self.line_number}\n" \
               f"| Word & Context: {self.word} {self.context}\n" \
               f"| Words: {self.words}\n" \
               f"| Embeddings: {self.embeddings}\n" \
               f"| Images: {self.images}\n" \
               f"\ Golden Image: {self.goldimg} \n"

    def __repr__(self):
        return self.__str__()
