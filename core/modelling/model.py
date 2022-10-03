import numpy
from core.variable import Inf


class ModelBase(object):

    _name = None

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

    def __init__(self, name) -> None:
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
                check_num = value.split('_')
                if check_num[-1].isnumeric():
                    number = int(float(check_num[-1]))
                    number += 1
                    number = str(number)
                    check_num[-1] = number
                    value = '_'.join(check_num)
                else:
                    check_num += ['1']
                    value = '_'.join(check_num)
            Model.models[value] = self
        self._name = value


class Port(ModelBase):

    def __init__(self, name) -> None:
        super().__init__(name)


class InPort(Port):

    def __init__(self, name='inport_0') -> None:
        super().__init__(name)
        self.signal = None

    def attach(self, signal):
        self.signal = signal
        

class OutPort(Port):

    def __init__(self, name='outport_0') -> None:
        super().__init__(name)
        self.signal = None

    def attach(self, signal):
        self.signal = signal


class Signal(ModelBase):

    def __init__(self, source, name='signal_0') -> None:
        super().__init__(name=name)
        self._source = source
        self._sinks = []
    
    def _propagate(self, value):
        for block in self._sinks:
            block._propagate(value)


class SignalMergeUnmerge(ModelBase):

    def __init__(self, name, n_inputs, n_outputs) -> None:
        super.__init__(name)
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._inports = {}
        self._listed_inports = []
        self._outports = {}
        self._listed_outports = []

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
            port = InPort(f'{self.name}/inport_{n}')
            self._inports[port.name] = port
            self._listed_inports.append(port)

    def create_outports(self, number):
        for n in range(number):
            port = OutPort(f'{self.name}/outport_{n}')
            self._inports[port.name] = port
            self._listed_outports.append(port)


class Mux(SignalMergeUnmerge):
    
    def __init__(self, sink, name='mux_0', n_inputs=2,) -> None:
        super().__init__(name, n_inputs, 1)

    def set_n_inputs(self, value):
        self._set_n_inputs(value)

    def _propagate(self, *v):
        out = numpy.array([i.value for i in self._listed_inports.signal.value])
        for port in self._listed_outports:
            port._propagate(out)


class DeMux(SignalMergeUnmerge):

    def __init__(self, name='demux_0', n_outputs=1) -> None:
        super().__init__(name, 1, n_outputs)

    def set_n_outputs(self, value):
        self._set_n_outputs(value)

    def _propagate(self, *v):
        out = self._inports[0]
        for port in self._listed_outports:
            port._propagate(out)  



class Block(ModelBase):

    """
    Use the port system to create conection points for other blocks. A signal object 
    """

    def __init__(self, name, from_file=None) -> None:
        self.sources = {}
        self.sinks = {}
        self.value = None
        super().__init__(name)

    def attach(self, block, as_source=True):
        if as_source == True:
            self.sources[block.name] = block
        else:
            self.sinks[block.name] = block

    def operate(self, value):
        self.value = value
        raise NotImplementedError()

    def _propagate(self, value):
        self.operate(value)
        for sink in self.sinks:
            sink.propagate(self.value)


class Model(ModelBase):

    """
    Represents a model object. 
    Can be a mathematical model or block diagram or state space representation.
    """

    def __init__(self, name) -> None:
        super().__init__(name)
