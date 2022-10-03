import cmath
import math
import warnings
import copy

import numpy


class NumericalIntegral(object):

    def __init__(self, eqn, limits) -> None:
        self._equation = None
        self._limits = None
        self._variables = set()
        self.equation = eqn
        self.limits = limits

    @property
    def equation(self):
        return self._equation

    @equation.setter
    def equation(self, val):
        if isinstance(val, Equation):
            self._equation = val

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, val):
        if isinstance(val, (list, tuple)):
            for v in val:
                print(v)
                self._variables.add(v[0])
            self._limits = val

    def integrate1D(self, var, equation, n_terms, method='trapz'):
        if isinstance(equation, (float, tuple)):
            equation = equation*(var[1][0] - var[1][1])

        elif var[0] not in equation.variables:
            equation = (var[1][0] - var[1][1])*equation
            if isinstance(equation, Equation):
                equation.reimplement_variables()
                equation.update_variables()
        elif method == 'trapz':
            h = (var[1][0] - var[1][1]) / n_terms
            divisions = [h*(i) for i in range(n_terms+1)]
            var_solution = []
            for n in range(len(divisions)):
                equation.variables[var[0]].value = divisions[n]
                if isinstance(equation, Equation):
                    sol = equation.solve()
                else:
                    sol = equation
                # print(sol)
                if n == 0 or n == len(divisions)-1:
                    
                    var_solution.append( sol  )
                else:
                    var_solution.append( 2*sol )
            solution = sum(var_solution) * h / 2
            # print(solution, 'sol')
            equation = solution
        elif method == 'simpson13':
            if n_terms % 2 != 0:
                n_terms += 1
            
            h = (var[1][0] - var[1][1]) / n_terms
            divisions = [h*(i) for i in range(n_terms+1)]
            var_solution = []
            for n in range(len(divisions)):
                equation.variables[var[0]].value = divisions[n]
                
                if isinstance(equation, Equation):
                    sol = equation.solve()
                else:
                    sol = equation
                # print(sol)
                if n == 0 or n == len(divisions) -1:

                    var_solution.append( sol  )
                else:
                    if n % 2 == 0:
                        var_solution.append( 2*sol )
                    elif n % 2 != 0:
                        var_solution.append( 4*sol )
            solution = sum(var_solution) * h / 3
            
            equation = solution
        elif method == 'simpson38':
            if n_terms % 3 != 0:
                n_terms = (n_terms // 3) * 3
            h = (var[1][0] - var[1][1]) / n_terms
            divisions = [h*(i) for i in range(n_terms+1)]
            print(var, equation.terms, h)
            # print(divisions)
            var_solution = []
            for n in range(len(divisions)):
                equation.variables[var[0]].value = divisions[n]
                if isinstance(equation, Equation):
                    sol = equation.solve()
                else:
                    sol = equation
                if n == 0 or n == len(divisions) -1:

                    var_solution.append( sol  )
                else:
                    if n % 3 == 0:
                        var_solution.append( 2*sol )
                    else:
                        var_solution.append( 3*sol )
            solution = 3 * sum(var_solution) * h / 8
            equation = solution
        
        return equation

    def integrate(self, n_terms=10, method='trapz', equation=None, limits=None):
        pass 
        

class Equation(object):

    def __init__(self, terms, operation, from_copy=False) -> None:
        self.variables = {}
        self.terms = terms
        if not from_copy:
            self.update_variables()
        self.value = None
        self.operation = operation
    
    def __str__(self) -> str:
        return repr(self)

    def update_variables(self):
        for i in self.terms:
            if isinstance(i, Variable):
                self.variables[i.name] = i
            elif isinstance(i, Equation):
                for j in i.variables:
                    self.variables[j] = i.variables[j]
    
    def reimplement_variables(self):
        for i in range(len(self.terms)):
            if isinstance(self.terms[i], Variable):
                self.terms[i] = self.variables[self.terms[i].name]
            elif isinstance(self.terms[i], Equation):
                for j in self.terms[i].variables:
                    self.terms[i].variables[j] = self.variables[j] 

    def __add__(self, other):
        if isinstance(self, Inf):
            return self
        elif isinstance(other, Inf):
            return other
        if other == 0:
            return self
        return Equation([self, other], '+')

    def __radd__(self, other):
        if isinstance(self, Inf):
            return self
        if other == 0:
            return self
        return Equation([other, self], '+')

    def __sub__(self, other):
        if isinstance(self, Inf):
            return self
        elif isinstance(other, Inf):
            return -other
        elif isinstance(other, Inf):
            other.ispos = not other.ispos
            return other
        if other == 0:
            return self
        return Equation([self, other], '-')
    
    def __rsub__(self, other):
        if isinstance(self, Inf):
            return -self
        if other == 0:
            return -self
        return Equation([other, self], '-')

    def __mul__(self, other):
        if isinstance(self, Inf):
            return self
        elif isinstance(other, Inf):
            return other
        if other == 0:
            return 0
        if other == 1:
            return self

        return Equation([self, other], '*')
    
    def __rmul__(self, other):
        if isinstance(self, Inf):
            if other < 0:
                self.ispos = not self.ispos
            return self
        if other == 0:
            return 0
        if other == 1:
            return self
        return Equation([other, self], '*')

    def __truediv__(self, other):
        if isinstance(self, Inf):
            return self
        elif isinstance(other, Inf):
            return 0
        return Equation([self, other], '/')

    def __rtruediv__(self, other):
        if isinstance(self, Inf):
            return 0
        if other == 0:
            return 0
        return Equation([other, self], '/')

    def __pow__(self, other):
        if isinstance(self, Inf):
            return self
        elif isinstance(other, Inf):
            return self
        if other == 0:
            return 1
        if other == 1:
            return self
        return Equation([self, other], '^')
    
    def __rpow__(self, other):
        if isinstance(self, Inf):
            if self.ispos:
                return self
            return 0
        if other == 0:
            return 0
        return Equation([other, self], '^')

    def __neg__(self):
        if isinstance(self, Inf):
            self.ispos = not self.ispos
            return self
        return -1*self

    def parse(self, _property, implicit=False):
        result = []
        if _property == 'name':
            for term in self.terms:
                if isinstance(term, Equation) and not isinstance(term, Variable):
                    r = term.parse(_property)
                else:
                    r = str(term)
                result.append(r)
            if not implicit:
                result = '('+f' {self.operation} '.join(result)+')'
            else:
                result = f'{self.operation}({"".join(result)})'
        return result

    def analyze(self, value):
        if self.operation == '+':
            return sum(value)
        elif self.operation == '-':
            return value[0]-value[1]
        elif self.operation == '*':
            return value[0]*value[1]
        elif self.operation == '/':
            try:
                return value[0]/value[1]
            except ZeroDivisionError:
                warnings.warn(f"Overflow with zero division in {'/'.join([str(i) for i in value])}")
                i = Inf()
                if isinstance(value[0], complex):
                    if value[0].real < 0:
                        i.ispos = not i.ispos
                elif value[0]<0:
                    i.ispos = not i.ispos
                return i
        elif self.operation == '^':
            return value[0]**value[1]

    def mdimsolve(self):
        pass

    def operate(self):
        operands = []

        for term in self.terms:
            if isinstance(term, (int, float, complex)):
                operands.append(term)
            else:
                term.operate()
                operands.append(term.value)
        if not operands:
            self.value = None
            return
        if all(i is not None for i in operands):
            self.value = self.analyze(operands)
        else:
            self.value = None

    def zeros_of_one_d_function(self, var, lim=1000):
        deqn = self.differentiate(var)
        x = self.variables[var] if isinstance(var, str) else var
        old = x.value
        x.value = complex(1, 0)
        v = 0
        i = 0
        while x.value != v and i <= lim:
            i += 1
            v = x.value
            x.value = x.value - self.solve() / deqn.solve()
        x.value = old
        return v

    def derivatives(self, v):
        solution = []
        if isinstance(v, str):
            solution.append(self.differentiate(v))
            return solution 
        elif isinstance(v, Variable):
            solution.append(self.differentiate(v.name))
            return solution
        else:
            for variable in v:
                if isinstance(variable, str):
                    solution.append(self.differentiate(variable))
                elif isinstance(variable, Variable):
                    solution.append(self.differentiate(variable.name))
        return solution

    def differentiate(self, i):
        if self.operation == '+' or self.operation == '-':
            result = []
            for operation in self.terms:
                if isinstance(operation, (int, float, complex)):
                    value = 0
                elif isinstance(operation, Equation) and i in operation.variables:
                    value = operation.differentiate(i)
                else:
                    value = 0
                result.append(value)
            result = Equation(result, self.operation)

        elif self.operation == '*':
            value = 0
            if (isinstance(self.terms[0], Equation) and i in self.terms[0].variables) or (isinstance(self.terms[1], Equation) and i in self.terms[1].variables):
                try:
                    d1 = self.terms[1].differentiate(i)
                except Exception as E:
                    d1 = 0
                finally:
                    pass
                
                try:
                    d2 = self.terms[0].differentiate(i)
                except Exception as E:
                    d2 = 0
                finally:
                    pass
                value = self.terms[0]*d1 + self.terms[1]*d2
            result = value
            
        elif self.operation == '/':
            value = 0
            if (isinstance(self.terms[0], Equation) and i in self.terms[0].variables) or (isinstance(self.terms[1], Equation) and i in self.terms[1].variables):
                try:
                    d1 = self.terms[1].differentiate(i)
                except Exception as E:
                    d1 = 0
                finally:
                    pass
                
                try:
                    d2 = self.terms[0].differentiate(i)
                except Exception as E:
                    d2 = 0
                finally:
                    pass
                
                value = (self.terms[1]*d2 - self.terms[0]*d1) / (self.terms[1] ** 2)
            result = value

        elif self.operation == '^':

            if (isinstance(self.terms[1], Equation) and i in self.terms[1].variables):
                # in the form a ** (f(x))
                new = Exp(self.terms[1]*Log(self.terms[0]))
                value = new.differentiate(i)
                result = value

            elif (isinstance(self.terms[0], Equation) and i in self.terms[0].variables) and (isinstance(self.terms[1], (int, float, complex)) or ((isinstance(self.terms[1], Equation) and i not in self.terms[1].variables))):
                # in the form f(x) ** c
                value = self.terms[1] * self.terms[0].differentiate(i) * self.terms[0] ** (self.terms[1]-1)
                result = value

            else:

                # in the form a ** b
                # unlikely case
                
                value = 0
                result = value
        return result

    def expand(self, start, stop, step=None, num=None):
        if step is not None:
            x = numpy.arange(start=start, stop=stop, step=step)
        elif num is not None:
            x, step = numpy.linspace(start, stop, num, retstep=True)
        vars_val = []
        for var in self.variables.values():
            vars_val.append(numpy.fromiter((self.set_and_solve(i, var) for i in x), numpy.float64))
        return step, x, numpy.array(*vars_val)

    def integrate(self, limits):
        return

    def set_and_solve(self, value, var):
        var.value = value
        return self.solve()

    def __repr__(self) -> str:
        return str(self.parse('name'))
    
    

    # def __deepcopy__(self, memo):
    #     # create a copy with self.linked_to *not copied*, just referenced.
    #     new = Equation(copy.deepcopy(self.terms, memo), copy.deepcopy(self.operation, memo), from_copy=True)
    #     new.variables = self.variables
    #     new.reimplement_variables()
    #     return new

    def solve(self):
        self.operate()
        return self.value

    def zero(self, var, start=0, max_i=1000):
        old_v = self.variables[var].value
        self.variables[var].value = start
        new_f = self.variables[var] - self / self.differentiate(var)
        # new_f.update_variables()
        run_i = 0
        old = 0
        while run_i < max_i:
            new_f.variables[var].value = new_f.solve()
            if new_f.variables[var].value == old:
                break
            run_i += 1
            old = new_f.variables[var].value
        self.variables[var].value = old_v
        return old


    def estimate(self, vars, *i):
        vars = [var.name if isinstance(var, Variable) else var for var in vars]
        result = []
        for items in zip(*i):
            for j in range(len(items)):
                self.variables[vars[j]].value = items[j]
            result += [self.solve()]
        return result

    

class Inf(Equation):

    def __init__(self) -> None:
        self.value = None
        self.ispos = True
        self.terms = []
        self.variables = {}
        self.operation = 'inf'

    def __bool__(self):
        return False

    def __eq__(self, __o: object) -> bool:
        return __o is None

    def __str__(self) -> str:
        return f'{"" if self.ispos else "-"}inf'

    def __repr__(self) -> str:
        return f'{str(self)}'


class Variable(Equation):

    INT = 'int'

    FLOAT = 'float'

    STR = 'str'

    def __init__(self, name, value=None, derivative_order=0,dtype='float') -> None:
        self.__name = ''
        self.variables = {name: self}
        self.name = name
        self.dtype = dtype
        self.value = value
        self.derivative_order = derivative_order
        Variable.add_variable(name, self)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if isinstance(name, str) and len(name) >= 1:
            self.__name = name
        else:
            raise ValueError('Invalid name for variable')

    @classmethod
    def add_variable(cls, name, value):
        pass
        # cls.user_variables[name] = value

    def __repr__(self) -> str:
        return f"<Variable '{self.__name}'>"

    def __str__(self) -> str:
        return self.__name

    def parse(self, *args):
        raise NotImplementedError('Parse not implemented for variable')
    
    def operate(self):
        pass

    def analyze(self, value):
        return self.value

    def differentiate(self, i):
        if i == self.name:
            return 1
        else:
            return 0


class Sin(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'sin')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.sin(value[0])
        return math.sin(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return Cos(self.terms[0]) * self.terms[0].differentiate(i)
        else:
            return 0
    

class Cos(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'cos')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')
    
    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.cos(value[0])
        return math.cos(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return -Sin(self.terms[0]) * self.terms[0].differentiate(i)
        else:
            return 0

class Tan(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'tan')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.tan(value[0])
        return math.tan(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return (self.terms[0].differentiate(i) / Cos(self.terms[0]))**2
        else:
            return 0

class Asin(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'asin')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.asin(value[0])
        return math.asin(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return self.terms[0].differentiate(i) / (1 - self.terms[0]**2)**0.5
        else:
            return 0

class Acos(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'acos')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.acos(value[0])
        return math.acos(value[0])
    
    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return -self.terms[0].differentiate(i) / (1 - self.terms[0]**2)**0.5
        else:
            return 0

class Atan(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'atan')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.atan(value[0])
        return math.atan(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return self.terms[0].differentiate(i) / (1 + self.terms[0]**2)
        else:
            return 0


class Exp(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'exp')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], Inf):
            if value[0].ispos:
                return Inf()
            else:
                return 0
        if isinstance(value[0], complex):
            return math.exp(value[0].real) * complex(math.cos(value[0].imag), math.sin(value[0].imag))
        return math.exp(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return self * self.terms[0].differentiate(i)
        else:
            return 0
    
    def integrateee(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return self * self.terms[0].integrate(i)
        else:
            return 0

class Log(Equation):

    def __init__(self, terms) -> None:
        super().__init__([terms], 'ln')

    def parse(self, _property, implicit=True):
        s = super().parse(_property, implicit)
        return s

    def __repr__(self) -> str:
        return self.parse('name')

    def analyze(self, value):
        if isinstance(value[0], complex):
            return cmath.log(value[0])
        return math.log(value[0])

    def differentiate(self, i):
        if i in self.variables and isinstance(self.terms[0], (Equation, Variable)):
            return self.terms[0].differentiate(i) / self.terms[0]
        else:
            return 0
    
    


if __name__ == '__main__':
    # Test
    x = Variable('x', 4)
    y = Variable('y', 3)
    k = Variable('k', 2)
    
    eqn = (0.5*x**2+4)
    eqn2 = eqn * (2*x)
    eqn3 = 2*x*eqn

    # print(eqn.solve())
    # print(eqn2.solve())
    # print(eqn2)
    # print(eqn3.solve())
    # print(eqn3)

    # print(eqn)
    # n = eqn.differentiate('x')
    print(eqn.solve())
    i = NumericalIntegral()
    S = i.integrate1D(n_terms=20, method='simpson38', var=('x', [0, 10]), equation=eqn)
    print(S)

    # eqn = eqn*(2*x)
    # print(eqn.solve())
    # i = eqn.integrate([('x', [1, 0])])
    # S = i.integrate(n_terms=20, method='simpson38')
    # print(S)
    # # u = n.differentiate('x')
    # print(u)
    
    # print(n, math.log(4))
    # print(eqn.expand(0, 1, 0.01))

    
    # print(u.differentiate('x').solve() + v.differentiate('y').solve())

    # print(d)
    # print(d.solve())
    # print((math.log(2)**2)*(math.log(math.log(2))+(1/math.log(2))))
