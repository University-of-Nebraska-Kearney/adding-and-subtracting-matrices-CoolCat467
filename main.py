"""Matrix - Matrix module for various applications."""

# Programmed by CoolCat467

from __future__ import annotations

# Matrix - Matrix module for various applications.
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
__version__ = "1.0.0"

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
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast, overload

if TYPE_CHECKING:
    from typing_extensions import Self


Number: TypeAlias = int | float  # | complex


def check_int_sequence(sequence: Sequence[object]) -> TypeGuard[Sequence[int]]:
    """Return True if all elements of given sequence are integers."""
    return all(isinstance(e, int) for e in sequence)


def check_int_float_sequence(
    sequence: Sequence[object],
) -> TypeGuard[Sequence[int | float]]:
    """Return True if all elements of given sequence are int or float."""
    return all(isinstance(e, (int, float)) for e in sequence)


@dataclass(repr=False, slots=True)
class Matrix(Sequence[Number]):
    """Matrix dataclass."""

    data: MutableSequence[Number]
    shape: tuple[int, int]

    def __repr__(self) -> str:
        """Return representation of this class."""
        # if 1 in self.shape:
        if self.row_count == 1:
            data_repr = repr(self.data)
        else:
            start, end = "(", ")"
            if type(self.data) not in (tuple, list):
                start = f"{self.data.__class__.__name__}(("
                end = "))"
            elif type(self.data) is list:
                start, end = "[", "]"
            data_repr = f"{start}\n"
            for row in self.iter_rows():
                data_repr += "  " + ", ".join(map(repr, row)) + ",\n"
            data_repr += end
        return f"{self.__class__.__name__}({data_repr}, shape={self.shape!r})"

    @classmethod
    def zeros(cls, width: int, height: int) -> Self:
        """Create matrix of zeros."""
        return cls(
            cast("MutableSequence[Number]", [0] * width * height),
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

    def __iter__(self) -> Iterator[Number]:
        """Yield matrix elements."""
        yield from self.data

    def __len__(self) -> int:
        """Return the number of elements in matrix."""
        return len(self.data)

    def _internal_data_from_iter(
        self,
        iterable: Iterable[Number],
    ) -> MutableSequence[Number]:
        """Return new internal data instance from iterable."""
        data_type = type(self.data)
        return data_type(iterable)  # type: ignore[call-arg]

    def _from_iter_new_shape(
        self,
        iterable: Iterable[Number],
        shape: tuple[int, int],
    ) -> Self:
        """Return new matrix from iterable and shape."""
        return self.__class__(
            self._internal_data_from_iter(iterable),
            shape=shape,
        )

    def _from_iter(self, iterable: Iterable[Number]) -> Self:
        """Return new matrix from iterable, maintaining current shape."""
        return self._from_iter_new_shape(
            iterable,
            shape=self.shape,
        )

    def _get_internal_index(self, row: int, column: int) -> int:
        """Return internal position for element at given row and column."""
        if row >= self.row_count:
            raise IndexError(
                f"row {row} is out of bounds for {self.shape} matrix",
            )
        if column >= self.column_count:
            raise IndexError(
                f"column {column} is out of bounds for {self.shape} matrix",
            )
        return self.row_count * row + column

    def _convert_slice_to_range(
        self,
        row: int | slice[int | None, int | None, int | None],
        column: int | slice[int | None, int | None, int | None],
    ) -> tuple[range, range]:
        """Return row and column ranges from associated slices / indexes."""
        if not isinstance(row, slice):
            if row < 0:
                row += self.row_count
            row = slice(row, row + 1)
        if not isinstance(column, slice):
            if column < 0:
                column += self.column_count
            column = slice(column, column + 1)
        return (
            range(*row.indices(self.row_count)),
            range(*column.indices(self.column_count)),
        )

    def _yield_internal_index_slice(
        self,
        row_range: range,
        column_range: range,
    ) -> Generator[int, None, None]:
        """Yield internal positions for row and column slices."""
        for row in row_range:
            for column in column_range:
                yield self._get_internal_index(row, column)

    @overload
    def __getitem__(self, index: int, /) -> Number: ...
    @overload
    def __getitem__(self, index: tuple[int, int], /) -> Number: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(
        self,
        index: slice[int | None, int | None, int | None],
        /,
    ) -> tuple[Number, ...]: ...
    @overload
    def __getitem__(
        self,
        index: tuple[
            int | slice[int | None, int | None, int | None],
            int | slice[int | None, int | None, int | None],
        ],
        /,
    ) -> Self: ...

    def __getitem__(
        self,
        index: int
        | tuple[int, int]
        | tuple[
            int | slice[int | None, int | None, int | None],
            int | slice[int | None, int | None, int | None],
        ]
        | slice[int | None, int | None, int | None],
        /,
    ) -> Number | tuple[Number, ...] | Self:
        """Return value at given index."""
        if isinstance(index, slice):
            return self._getitem_slice(index)
        if isinstance(index, tuple):
            row, column = index
            if isinstance(row, int) and isinstance(column, int):
                return self.data[self._get_internal_index(row, column)]
            row_range, column_range = self._convert_slice_to_range(row, column)
            return self._from_iter_new_shape(
                (
                    self.data[i]
                    for i in self._yield_internal_index_slice(
                        row_range,
                        column_range,
                    )
                ),
                shape=(len(row_range), len(column_range)),
            )
        return self.data[index]

    def _getitem_slice(
        self,
        slice_: slice[int | None, int | None, int | None],
    ) -> tuple[Number, ...]:
        """Return dynamic array of slice elements."""
        slice_range = range(*slice_.indices(len(self)))
        return tuple(self.data[i] for i in slice_range)

    @overload
    def __setitem__(
        self,
        index: int | tuple[int, int],
        value: Number,
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        index: slice[int | None, int | None, int | None],
        value: Iterable[Number],
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        index: tuple[
            int | slice[int | None, int | None, int | None],
            int | slice[int | None, int | None, int | None],
        ],
        value: Iterable[Number],
        /,
    ) -> None: ...

    def __setitem__(
        self,
        index: int
        | tuple[int, int]
        | tuple[
            int | slice[int | None, int | None, int | None],
            int | slice[int | None, int | None, int | None],
        ]
        | slice[int | None, int | None, int | None],
        value: Number | Iterable[Number],
        /,
    ) -> None:
        """Set item at given index to given value."""
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise ValueError("slice set item value must be iterable")
            self._setitem_slice(index, value)
            return
        if isinstance(index, tuple):
            row, column = index
            if isinstance(row, int) and isinstance(column, int):
                assert not isinstance(value, Iterable)
                self.data[self._get_internal_index(row, column)] = value
                return
            row_range, column_range = self._convert_slice_to_range(row, column)
            try:
                assert isinstance(value, Iterable)
                for i, v in zip(
                    self._yield_internal_index_slice(row_range, column_range),
                    value,
                    strict=True,
                ):
                    self.data[i] = v
            except ValueError as exc:
                raise ValueError(
                    "slice element count is not the same size as new element iterable",
                ) from exc
        else:
            assert not isinstance(value, Iterable)
            self.data[index] = value

    def _setitem_slice(
        self,
        slice_: slice[int | None, int | None, int | None],
        iterable: Iterable[Number],
    ) -> None:
        """Overwrite multiple elements at once."""
        for index, item in zip(
            range(*slice_.indices(len(self))),
            iterable,
            strict=True,
        ):
            self.data[index] = item

    def iter_rows(self) -> Generator[MutableSequence[Number], None, None]:
        """Yield matrix rows."""
        for row in range(self.row_count):
            yield self.data[
                row * self.column_count : (row + 1) * self.column_count
            ]

    def iter_columns(
        self,
    ) -> Generator[MutableSequence[Number], None, None]:
        """Yield matrix columns."""
        for col in range(self.column_count):
            yield self.data[col :: self.column_count]

    def iter_rows_vector(self) -> Generator[Self, None, None]:
        """Yield matrix rows as vectors."""
        for row_elements in self.iter_rows():
            yield self._from_iter_new_shape(
                row_elements,
                shape=(1, self.column_count),
            )

    def iter_columns_vector(self) -> Generator[Self, None, None]:
        """Yield matrix columns as vectors."""
        for column_elements in self.iter_columns():
            yield self._from_iter_new_shape(
                column_elements,
                shape=(self.row_count, 1),
            )

    @property
    def rows(self) -> tuple[Self, ...]:
        """Matrix rows."""
        return tuple(self.iter_rows_vector())

    @property
    def columns(self) -> tuple[Self, ...]:
        """Matrix columns."""
        return tuple(self.iter_columns_vector())

    def __round__(self, ndigits: int | None = None) -> Self:
        """Return round of all elements."""
        assert check_int_float_sequence(self.data)
        return self._from_iter(round(e, ndigits) for e in self.data)

    def __mul__(self, rhs: int) -> Self:
        """Return scalar multiply by rhs."""
        return self._from_iter(e * rhs for e in self.data)

    def __abs__(self) -> Self:
        """Return absolute value of all elements."""
        return self._from_iter(map(abs, self.data))

    def __ceil__(self) -> Self:
        """Return ceiling of all elements."""
        assert check_int_float_sequence(self.data)
        return self._from_iter(map(ceil, self.data))

    def __floor__(self) -> Self:
        """Return ceiling of all elements."""
        assert check_int_float_sequence(self.data)
        return self._from_iter(map(floor, self.data))

    def __trunc__(self) -> Self:
        """Return truncate of all elements."""
        assert check_int_float_sequence(self.data)
        return self._from_iter(map(trunc, self.data))

    def __invert__(self) -> Self:
        """Return bitwise invert of all elements."""
        if not check_int_sequence(self.data):
            raise ValueError(
                "All elements of {self.__class__.__name__} must be integers for this operation.",
            )
        return self._from_iter(~e for e in self.data)

    def __pos__(self) -> Self:
        """Return positive of all elements."""
        return self._from_iter(+e for e in self.data)

    def __neg__(self) -> Self:
        """Return negation of all elements."""
        return self._from_iter(-e for e in self.data)

    @staticmethod
    def _scalar_op_decorator(
        map_function: Callable[[Number, Number], Number],
    ) -> Callable[
        [Matrix, Number | Iterable[Number] | Matrix],
        Matrix,
    ]:
        """Return wrapper for scalar operation."""

        def wrapper(
            self: Matrix,
            rhs: Number | Iterable[Number] | Matrix,
        ) -> Matrix:
            # If rhs is just a number, do map with that same value for
            # every element
            if not isinstance(rhs, Iterable):
                return self._from_iter(map_function(e, rhs) for e in self)
            # Otherwise, check if matrix if matrix check sizes.
            if isinstance(rhs, Matrix) and rhs.shape != self.shape:
                raise ValueError(
                    f"Shape of right hand side matrix {rhs.shape} does "
                    f"not match own shape {self.shape}",
                )
            return self._from_iter(
                map_function(e, rhse)
                for e, rhse in zip(self, rhs, strict=True)
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
            if not check_int_sequence(self.data):
                raise ValueError(
                    "All elements of {self.__class__.__name__} must be integers for this operation.",
                )
            # If rhs is just a number, do map with that same value for
            # every element
            if not isinstance(rhs, Iterable):
                return self._from_iter(map_function(e, rhs) for e in self.data)

            rhs_int: tuple[Number, ...] = tuple(rhs)
            if not check_int_sequence(rhs_int):
                raise ValueError(
                    "All elements of right hand side must be integers for this operation.",
                )

            # Otherwise, check if matrix if matrix check sizes.
            if isinstance(rhs, Matrix) and rhs.shape != self.shape:
                raise ValueError(
                    f"Shape of right hand side matrix {rhs.shape} does "
                    f"not match own shape {self.shape}",
                )
            return self._from_iter(
                map_function(e, rhse)
                for e, rhse in zip(self.data, rhs_int, strict=True)
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
    def __add__(x: Number, y: Number) -> Number:  # noqa: N805
        """Return scalar add by rhs."""
        return x + y

    @_scalar_op_decorator
    def __sub__(x: Number, y: Number) -> Number:  # noqa: N805
        """Return scalar subtract by rhs."""
        return x - y

    @_scalar_op_decorator
    def __truediv__(x: Number, y: Number) -> Number:  # noqa: N805
        """Return scalar floating point division by rhs."""
        return x / y

    @_scalar_op_decorator
    def __floordiv__(x: Number, y: Number) -> Number:  # noqa: N805
        """Return scalar floor division by rhs."""
        return x // y

    @_scalar_op_decorator
    def __mod__(x: Number, y: Number) -> Number:  # noqa: N805
        """Return scalar modulo by rhs."""
        return x % y

    @_scalar_op_decorator
    def __pow__(
        x: Number,  # noqa: N805
        y: Number,
        # mod: Number | None = None,
    ) -> Number:
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

        return self._from_iter_new_shape(
            (
                sumprod(row, column)
                for column in rhs.iter_columns()
                for row in self.iter_rows()
            ),
            (self.row_count, rhs.column_count),
        )

    def transpose(self) -> Self:
        """Return transpose of matrix."""
        return self._from_iter_new_shape(
            (element for column in self.iter_columns() for element in column),
            shape=(self.column_count, self.row_count),
        )

    @property
    def T(self) -> Self:  # noqa: N802
        """Transpose."""
        return self.transpose()

    def _yield_minor_elements(
        self,
        remove_row: int,
        remove_column: int,
    ) -> Generator[Number, None, None]:
        """Yield elements of minor matrix given row and column to remove."""
        for row_index, row in enumerate(self.iter_rows()):
            if row_index == remove_row:
                continue
            for column_index, element in enumerate(row):
                if column_index == remove_column:
                    continue
                yield element

    def minor(self, index: tuple[int, int]) -> Self:
        """Return new matrix without index location."""
        return self._from_iter_new_shape(
            self._yield_minor_elements(*index),
            (self.row_count - 1, self.column_count - 1),
        )

    def determinent(self) -> Number:
        """Return the determinent of this matrix."""
        if self.shape == (1, 1):
            return self[0, 0]
        if self.row_count != self.column_count:
            raise ValueError(f"{self.shape} matrix is not a square matrix")
        value: Number = 0
        adding = True
        for y in range(self.row_count):
            value += (
                ((adding << 1) - 1)
                * self.minor((0, y)).determinent()
                * self[0, y]
            )
            adding = not adding
        return value

    def get_pos_cofactor(self, index: tuple[int, int]) -> Number:
        """Return cofactor of item at index in this matrix."""
        return ((((sum(index) & 1) ^ 1) << 1) - 1) * self.minor(
            index,
        ).determinent()

    def cofactor(self) -> Matrix:
        """Return cofactor of self."""
        return self._from_iter(
            self.get_pos_cofactor((r, c))
            for r in range(self.row_count)
            for c in range(self.column_count)
        )

    def adjugate(self) -> Matrix:
        """Return adjugate of self."""
        return self.cofactor().transpose()

    adjoint = adjugate

    def inverse(self) -> Matrix:
        """Return the inverse of this matrix."""
        det = self.determinent()
        if det == 0:
            raise ZeroDivisionError("Determinent of this matrix is zero")
        return self.adjugate() / det


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
    """Return matrix from user input.

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
