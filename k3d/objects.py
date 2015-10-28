class Objects:
    def __init__(self, output):
        self.__items = []
        self.__output = output
        self.__strategy = self.__buffer

    def add(self, item):
        self.__strategy(item)
        pass

    def flush(self):
        self.__flush_items()
        self.__strategy = self.__output

    def __flush_items(self):
        while len(self.__items):
            self.__output(self.__items.pop(0))

    def __buffer(self, obj):
        self.__items.append(obj)
