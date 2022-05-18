from core.modelling.model import Block


class Add(Block):
    
    def __init__(self, name='Add', from_file=None) -> None:
        super().__init__(name, from_file)