from core.modelling.model import InPort, OutPort, Signal, Mux, DeMux
from core.variable import Variable, Exp, Sin
from core.approximations.taylor import taylor1D


x = Variable('x')

eqn = Exp(2*x) + Sin(x+10)
taylorapprox = taylor1D(eqn, x, 6, 9)
x.value = 6
print(eqn.solve(), taylorapprox.solve())