from core.modelling.model import InPort, OutPort, Signal, Mux, DeMux, ModelBase, Add
from core.variable import Variable, Exp, Sin
from core.approximations.taylor import taylor1D


inport = InPort()

inport2 = InPort()

inport3 = InPort()

inport4 = InPort()

print(ModelBase.models)