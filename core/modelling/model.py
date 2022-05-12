import numpy
from core.variable import Inf


class Port(object):

    def __init__(self, name) -> None:
        self._name = name
        self._value = Inf()


class InPort(Port):

    def __init__(self, name) -> None:
        super().__init__(name)


class OutPort(Port):

    def __init__(self, name) -> None:
        super().__init__(name)


class Signal(object):

    def __init__(self, name, source) -> None:
        self._name = name
        self._source = source
        self._sinks = []
    
    def push(self, value):
        for port in self._sinks:
            port.value = value


class SignalMergeUnmerge(object):

    def __init__(self, n_inputs, n_outputs) -> None:
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._inports = {}
        self._outports = {}

    def _set_n_inputs(self, value):
        self.n_outputs = value

    def _set_n_outputs(self, value):
        self.n_outputs = value

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

    def create_inports(self, number):
        for n in range(number):
            port = InPort(f'port_{n}')
            self._inports[port.name] = port

    def create_outports(self, number):
        for n in range(number):
            port = OutPort(f'port_{n}')
            self._inports[port.name] = port


class Mux(SignalMergeUnmerge):
    
    def __init__(self, n_inputs=2, n_outputs=1) -> None:
        super().__init__(n_inputs, n_outputs)

    def set_n_inputs(self, value):
        self._set_n_inputs(value)

    def push(self):
        out = numpy.array([i.value for i in self._inports])


class DeMux(object):

    def __init__(self, n_inputs=2, n_outputs=1) -> None:
        super().__init__(n_inputs, n_outputs)

    def set_n_outputs(self, value):
        self._set_n_outputs(value)


class Block(object):

    sources = None

    sinks = None

    """
    Use the port system to create conection points for other blocks. A signal object 
    """

    def __init__(self, name, from_file=None) -> None:
        pass


class Model(object):

    """
    Represents a model object. 
    Can be a mathematical model or block diagram or state space representation.
    """


    models = {}
    """
    Stores all modeled objects with names as keys. No two models can be made the same. 
    """
    
    @classmethod
    def check_named(cls, name):
        """
        Class checks if a model name is the same as another model.
        """
        return name in cls.models

    def __init__(self, name, ) -> None:
        self._name = None
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("Model name must be a string.")
        if not Model.check_named(value):
            Model.models[value] = self
        else:
            while Model.check_named(value):
                value = value + '_1'
            Model.models[value] = self
        self._name = value
