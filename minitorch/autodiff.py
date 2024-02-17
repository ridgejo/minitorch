from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from .operators import map
# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    x_plus = list(vals)
    x_min = list(vals)
    x_plus[arg] = x_plus[arg] + epsilon
    x_min[arg] = x_min[arg] - epsilon
    f_prime = (f(*x_plus) - f(*x_min)) / (2 * epsilon)
    return f_prime


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    order = []
    visited = set()

    def visit(var):
        if var.unique_id in visited:
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        visited.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # Call topological sort to get an ordered queue
    # Create a dictionary of Scalars and current derivatives
    # For each node in backward order, pull a completed Scalar and derivative from the queue: 
    #     a. if the Scalar is a leaf, add its final derivative (accumulate_derivative) and loop to (1) 
    #     b. if the Scalar is not a leaf, 
    #         1) call .chain_rule on the last function with d_out
    #         2) loop through all the Scalars+derivative produced by the chain rule 
    #         3) accumulate derivatives for the Scalar in a dictionary
    order = topological_sort(variable)
    s_d_dict = {s.unique_id: s.derivative if not s.derivative is None else 0.0 for s in order}
    s_d_dict[variable.unique_id] = deriv
    for var in order:
        if var.is_leaf():
            var.accumulate_derivative(s_d_dict[var.unique_id])
        else:
            chain = var.chain_rule(s_d_dict[var.unique_id])
            for s, d in chain:
                s_d_dict[s.unique_id] += d

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
