"""Matrix - Matrix module for various applications."""

# Programmed by CoolCat467

from __future__ import annotations

# Vector - Vector module for various applications.
# Copyright (C) 2026  CoolCat467
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__title__ = "Matrix Module"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "0.0.0"

from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass
from functools import update_wrapper
from math import ceil, floor, sumprod, trunc
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(repr=False, slots=True)
class Matrix(Sequence[int | float]):
    """Matrix dataclass."""

    data: MutableSequence[int | float]
    shape: tuple[int, int]

    def __repr__(self) -> str:
        """Return representation of this class."""
        # if 1 in self.shape:
        if self.row_count == 1:
            data_repr = repr(self.data)
        else:
            data_repr = "[\n"
            for row in self.iter_rows():
                data_repr += "  " + ", ".join(map(repr, row)) + ",\n"
            data_repr += "]"
        return f"{self.__class__.__name__}({data_repr}, shape={self.shape!r})"

    @classmethod
    def zeros(cls, width: int, height: int) -> Self:
        """Create matrix of zeros."""
        return cls(
            cast("MutableSequence[int | float]", [0] * width * height),
            shape=(width, height),
        )

    @classmethod
    def identity(cls, n: int) -> Self:
        """Create an n x n identity matrix."""
        self = cls.zeros(n, n)
        for xy in range(n):
            self[xy, xy] = 1
        return self

    @property
    def row_count(self) -> int:
        """Number of rows."""
        return self.shape[0]

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return self.shape[1]

    def __iter__(self) -> Iterator[int | float]:
        """Yield matrix elements."""
        yield from self.data

    def __len__(self) -> int:
        """Return the number of elements in matrix."""
        return len(self.data)

    @overload
    def __getitem__(self, index: int | tuple[int, int], /) -> int | float: ...
    @overload
    def __getitem__(
        self,
        index: slice[int | None, int | None, int | None],
        /,
    ) -> tuple[int | float, ...]: ...

    def __getitem__(
        self,
        index: int
        | tuple[int, int]
        | slice[int | None, int | None, int | None],
        /,
    ) -> int | float | tuple[int | float, ...]:
        """Return value at given index."""
        if isinstance(index, slice):
            return self._getitem_slice(index)
        if isinstance(index, tuple):
            row, column = index
            # if isinstance(row, int) and isinstance(column, int):
            return self.data[self.row_count * row + column]
        return self.data[index]

    def _getitem_slice(
        self,
        slice_: slice[int | None, int | None, int | None],
    ) -> tuple[int | float, ...]:
        """Return dynamic array of slice elements."""
        slice_range = range(*slice_.indices(len(self)))
        return tuple(self[x] for x in slice_range)

    @overload
    def __setitem__(
        self,
        index: int | tuple[int, int],
        value: int | float,
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        index: slice[int | None, int | None, int | None],
        value: Iterable[int | float],
        /,
    ) -> None: ...

    def __setitem__(
        self,
        index: int
        | tuple[int, int]
        | slice[int | None, int | None, int | None],
        value: int | float | Iterable[int | float],
        /,
    ) -> None:
        """Set item at given index to given value."""
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise ValueError("slice set item value must be iterable")
            self._setitem_slice(index, value)
            return
        assert not isinstance(value, Iterable)
        if isinstance(index, tuple):
            row, column = index
            self.data[self.row_count * row + column] = value
        else:
            self.data[index] = value

    def _setitem_slice(
        self,
        slice_: slice[int | None, int | None, int | None],
        iterable: Iterable[int | float],
    ) -> None:
        """Overwrite multiple elements at once."""
        for index, item in zip(
            range(*slice_.indices(len(self))),
            iterable,
            strict=True,
        ):
            self[index] = item

    def iter_rows(self) -> Generator[MutableSequence[int | float], None, None]:
        """Yield matrix rows."""
        for row in range(self.row_count):
            yield self.data[
                row * self.column_count : (row + 1) * self.column_count
            ]

    def iter_columns(
        self,
    ) -> Generator[MutableSequence[int | float], None, None]:
        """Yield matrix columns."""
        for col in range(self.column_count):
            yield self.data[col :: self.column_count]

    def iter_rows_vector(self) -> Generator[Self, None, None]:
        """Yield matrix rows as vectors."""
        for row_elements in self.iter_rows():
            yield self.__class__(row_elements, shape=(1, self.column_count))

    def iter_columns_vector(self) -> Generator[Self, None, None]:
        """Yield matrix columns as vectors."""
        for column_elements in self.iter_columns():
            yield self.__class__(column_elements, shape=(self.row_count, 1))

    @property
    def rows(self) -> tuple[Self, ...]:
        """Matrix rows."""
        return tuple(self.iter_rows_vector())

    @property
    def columns(self) -> tuple[Self, ...]:
        """Matrix columns."""
        return tuple(self.iter_columns_vector())

    def __mul__(self, rhs: int) -> Self:
        """Return scalar multiply by rhs."""
        return self.__class__([e * rhs for e in self], shape=self.shape)

    def __abs__(self) -> Self:
        """Return absolute value of all elements."""
        return self.__class__(list(map(abs, self)), shape=self.shape)

    def __ceil__(self) -> Self:
        """Return ceiling of all elements."""
        return self.__class__(list(map(ceil, self)), shape=self.shape)

    def __floor__(self) -> Self:
        """Return ceiling of all elements."""
        return self.__class__(list(map(floor, self)), shape=self.shape)

    def __trunc__(self) -> Self:
        """Return truncate of all elements."""
        return self.__class__(list(map(trunc, self)), shape=self.shape)

    def __invert__(self) -> Self:
        """Return bitwise invert of all elements."""
        assert all(isinstance(e, int) for e in self), (
            "Invert does not work on floats."
        )
        return self.__class__(
            [~e for e in cast("Iterable[int]", self)],
            shape=self.shape,
        )

    def __pos__(self) -> Self:
        """Return positive of all elements."""
        return self.__class__([+e for e in self], shape=self.shape)

    def __neg__(self) -> Self:
        """Return negation of all elements."""
        return self.__class__([-e for e in self], shape=self.shape)

    @staticmethod
    def _scalar_op_decorator(
        map_function: Callable[[int | float, int | float], int | float],
    ) -> Callable[
        [Matrix, int | float | Iterable[int | float] | Matrix],
        Matrix,
    ]:
        """Return wrapper for scalar operation."""

        def wrapper(
            self: Matrix,
            rhs: int | float | Iterable[int | float] | Matrix,
        ) -> Matrix:
            # If rhs is just a number, do map with that same value for every element
            if not isinstance(rhs, Iterable):
                return self.__class__(
                    [map_function(e, rhs) for e in self],
                    shape=self.shape,
                )
            # Otherwise, check if matrix if matrix check sizes.
            if isinstance(rhs, Matrix) and rhs.shape != self.shape:
                raise ValueError(
                    f"Shape of right hand side matrix {rhs.shape} does not match own shape {self.shape}",
                )
            return self.__class__(
                [
                    map_function(e, rhse)
                    for e, rhse in zip(self, rhs, strict=True)
                ],
                shape=self.shape,
            )

        # Not using `functools.wraps` because something is not correct
        # about setting `assigned` and working with mypy. Need to make
        # sure that annotations of wrapper function are maintained and
        # not overridden by wrapped map function.
        update_wrapper(
            wrapper,
            map_function,
            assigned=("__module__", "__name__", "__qualname__", "__doc__"),
        )
        return wrapper

    @staticmethod
    def _scalar_int_op_decorator(
        map_function: Callable[[int, int], int],
    ) -> Callable[[Matrix, int | Iterable[int] | Matrix], Matrix]:
        """Return wrapper for integer scalar operation."""

        def wrapper(self: Matrix, rhs: int | Iterable[int] | Matrix) -> Matrix:
            # Make sure all items of matrix are integers
            for e in self:
                if not isinstance(e, int):
                    raise ValueError(
                        "All elements of matrix must be integers for this operation.",
                    )
            self_int = cast("Iterable[int]", self)
            # If rhs is just a number, do map with that same value for every element
            if not isinstance(rhs, Iterable):
                return self.__class__(
                    [map_function(e, rhs) for e in self_int],
                    shape=self.shape,
                )

            rhs_int: list[int] = []
            for e in rhs:
                if not isinstance(e, int):
                    raise ValueError(
                        "All elements of right hand side must be integers for this operation.",
                    )
                rhs_int.append(e)

            # Otherwise, check if matrix if matrix check sizes.
            if isinstance(rhs, Matrix) and rhs.shape != self.shape:
                raise ValueError(
                    f"Shape of right hand side matrix {rhs.shape} does not match own shape {self.shape}",
                )
            return self.__class__(
                [
                    map_function(e, rhse)
                    for e, rhse in zip(self_int, rhs_int, strict=True)
                ],
                shape=self.shape,
            )

        # Not using `functools.wraps` because something is not correct
        # about setting `assigned` and working with mypy. Need to make
        # sure that annotations of wrapper function are maintained and
        # not overridden by wrapped map function.
        update_wrapper(
            wrapper,
            map_function,
            assigned=("__module__", "__name__", "__qualname__", "__doc__"),
        )
        return wrapper

    @_scalar_op_decorator
    def __add__(x: int | float, y: int | float) -> int | float:  # noqa: N805
        """Return scalar add by rhs."""
        return x + y

    @_scalar_op_decorator
    def __sub__(x: int | float, y: int | float) -> int | float:  # noqa: N805
        """Return scalar subtract by rhs."""
        return x - y

    @_scalar_op_decorator
    def __truediv__(x: int | float, y: int | float) -> int | float:  # noqa: N805
        """Return scalar floating point division by rhs."""
        return x / y

    @_scalar_op_decorator
    def __floordiv__(x: int | float, y: int | float) -> int | float:  # noqa: N805
        """Return scalar floor division by rhs."""
        return x // y

    @_scalar_op_decorator
    def __mod__(x: int | float, y: int | float) -> int | float:  # noqa: N805
        """Return scalar modulo by rhs."""
        return x % y

    @_scalar_op_decorator
    def __pow__(
        x: int | float,  # noqa: N805
        y: int | float,
        # mod: int | float | None = None,
    ) -> int | float:
        """Return scalar modulo by rhs."""
        return pow(x, y)  # , mod)

    @_scalar_int_op_decorator
    def __and__(x: int, y: int) -> int:  # noqa: N805
        """Return scalar bitwise and by rhs."""
        return x & y

    @_scalar_int_op_decorator
    def __or__(x: int, y: int) -> int:  # noqa: N805
        """Return scalar bitwise or by rhs."""
        return x | y

    @_scalar_int_op_decorator
    def __xor__(x: int, y: int) -> int:  # noqa: N805
        """Return scalar bitwise xor by rhs."""
        return x ^ y

    @_scalar_int_op_decorator
    def __lshift__(x: int, y: int) -> int:  # noqa: N805
        """Return scalar left shift by rhs."""
        return x << y

    @_scalar_int_op_decorator
    def __rshift__(x: int, y: int) -> int:  # noqa: N805
        """Return scalar right shift by rhs."""
        return x >> y

    def __matmul__(self, rhs: Matrix) -> Self:
        """Return matrix multiply by rhs matrix."""
        if self.column_count != rhs.row_count:
            raise ValueError(
                f"Cannot multiply {self.shape} matrix by {rhs.shape} matrix.",
            )

        return self.__class__(
            [
                sumprod(row, column)
                for column in rhs.iter_columns()
                for row in self.iter_rows()
            ],
            (self.row_count, rhs.column_count),
        )

    @property
    def T(self) -> Self:  # noqa: N802
        """Transpose."""
        return self.__class__(
            [element for column in self.iter_columns() for element in column],
            shape=(self.column_count, self.row_count),
        )


def ask_int(prompt: str | None = None) -> int:
    """Prompt user for integer input."""
    real_prompt = prompt + " > " if prompt is not None else ""
    while True:
        try:
            return int(input(real_prompt))
        except ValueError:
            print("Input is not an integer, please try again.\n")


def ask_float(prompt: str | None = None) -> float:
    """Prompt user for float input."""
    real_prompt = prompt + " > " if prompt is not None else ""
    while True:
        try:
            return float(input(real_prompt))
        except ValueError:
            print("Input is not a float, please try again.\n")


def get_matrix() -> Matrix:
    """Create matrix from user input.

    Creates and returns a matrix from user input such that the shape
    and dimension is determined by the user.

    This should be a hybrid of the fixed-sized and the sentinel model we
    discussed in class. The user should be prompted for the number or
    rows and columns before entering values to fill the matrix.
    """
    rows = ask_int("Row count")
    cols = ask_int("Column count")
    elements: list[int | float] = []
    for y in range(cols):
        for x in range(rows):
            value = ask_float(f"Element {x} {y}")
            if int(value) == value:
                value = int(value)
            elements.append(value)
    return Matrix(elements, shape=(rows, cols))


def add_matrix(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """Add two matrices.

    Takes two matrices as parameters, verifies that they can be added
    together, adds them together, and returns the sum.
    """
    return matrix1 + matrix2


def test() -> None:
    """Test."""
    ##A = Matrix(data=[0, 0, 2, 1, 3, -2, 1, -2, 1], shape=(3, 3))
    ##X = Matrix(data=[1, -2, 3], shape=(3, 1))
    ###print(f"{A = }")
    ###print(f"{X = }")
    ##print(f'{tuple(A @ X) == (6, -11, 8)=}')
    ##
    ##A = Matrix(data=[0, 0, 2, 1, 3, -2, 1, -2, 1], shape=(3, 3))
    ##X = Matrix(data=[-1.1, 1.7, 4.5], shape=(3, 1))
    ##print(f'{tuple(A @ X) == (9, -5, 0)=}')

    print("Matrix 1")
    m1 = get_matrix()
    print()
    print(m1)
    print("\nMatrix 2")
    m2 = get_matrix()
    print()
    print(m2)
    print("\nSum")
    sum_ = add_matrix(m1, m2)
    print(sum_)


if __name__ == "__main__":  # pragma: nocover
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")
    test()
