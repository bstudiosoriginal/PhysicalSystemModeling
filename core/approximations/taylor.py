from core.variable import Equation
from math import factorial


def taylor1D(equation, variable, point=0, steps=4):
    if isinstance(equation, Equation):
        polynomials = []
        for n in range(steps):
            if isinstance(variable, str):
                variable = equation.variables[variable]
            variable.value = point
            solution = equation.solve()
            polynomial = (solution / factorial(n)) * (variable - point)**n
            polynomials.append(polynomial)
            equation  = equation.differentiate(variable.name)
        return sum(polynomials) if polynomials else 0


