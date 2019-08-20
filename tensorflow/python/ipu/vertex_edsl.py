# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from enum import IntEnum


class BaseType(IntEnum):
  Unknown = 0
  Int16 = 1
  Int32 = 2
  Float16 = 3
  Float32 = 4
  Sequence = 5


class Type:
  def __init__(self, type):
    self._type = type

  def getType(self):
    return self._type

  def getElementType(self):
    return self.getType()


class Unknown(Type):
  def __init__(self):
    Type.__init__(self, BaseType.Unknown)


class Int16(Type):
  def __init__(self):
    Type.__init__(self, BaseType.Int16)


class Int32(Type):
  def __init__(self):
    Type.__init__(self, BaseType.Int32)


class Float16(Type):
  def __init__(self):
    Type.__init__(self, BaseType.Float16)


class Float32(Type):
  def __init__(self):
    Type.__init__(self, BaseType.Float32)


class Sequence(Type):
  def __init__(self, element_type):
    Type.__init__(self, BaseType.Sequence)
    self._element_type = element_type

  def getElementType(self):
    return self._element_type.getElementType()


class DefaultNameSource:
  def __init__(self):
    self._i = 0

  def GetNextName(self):
    self._i += 1
    return self._i - 1


class VertexExpression:
  def __init__(self, type):
    self._type = type

  def __abs__(self):
    return UnaryFunctionExpr("std::abs", self)

  def __add__(self, other):
    return BinaryExpr("+", self, other)

  def __sub__(self, other):
    return BinaryExpr("-", self, other)

  def __mul__(self, other):
    return BinaryExpr("*", self, other)

  def __neg__(self):
    return BinaryExpr("-", ConstantVertexExpr(0), self)

  def __invert__(self):
    return self.__neg__()

  def __pos__(self):
    return self

  def getType(self):
    return self._type

  def __str__(self):
    raise NotImplementedError()

  def lower(self, source):
    raise NotImplementedError()


class ConstantVertexExpr(VertexExpression):
  def __init__(self, val):
    if isinstance(val, int):
      VertexExpression.__init__(self, Int32())
    elif isinstance(val, float):
      VertexExpression.__init__(self, Float32())
    else:
      raise NotImplementedError()
    self._val = val

  def __str__(self):
    return str(self._val)

  def lower(self, source):
    return "(" + str(self._val) + ")"


def constant(value):
  return ConstantVertexExpr(value)


class PlaceholderVertexExpr(VertexExpression):
  def __init__(self, name, type):
    VertexExpression.__init__(self, type)
    self._name = name

  def __str__(self):
    return str(self._name)

  def lower(self, source):
    return self._name


class BinaryExpr(VertexExpression):
  def __init__(self, op, lhs, rhs):
    VertexExpression.__init__(self, lhs.getType())
    self._op = op
    self._lhs = lhs
    self._rhs = rhs

  def __str__(self):
    return "(" + str(self._lhs) + str(self._op) + str(self._rhs) + ")"

  def lower(self, source):
    return "(" + self._lhs.lower(source) + str(
        self._op) + self._rhs.lower(source) + ")"


class BinaryFunctionExpr(VertexExpression):
  def __init__(self, op, lhs, rhs):
    VertexExpression.__init__(self, lhs.getType())
    self._op = op
    self._lhs = lhs
    self._rhs = rhs

  def __str__(self):
    return "(" + str(self._op) + "(" + str(self._lhs) + "," + str(
        self._rhs) + "))"

  def lower(self, source):
    return "(" + str(self._op) + "(" + str(
        self._lhs.lower(source)) + "," + str(self._rhs.lower(source)) + "))"


class UnaryFunctionExpr(VertexExpression):
  def __init__(self, op, lhs):
    VertexExpression.__init__(self, lhs.getType())
    self._op = op
    self._lhs = lhs

  def __str__(self):
    return "(" + str(self._op) + "(" + str(self._lhs) + ") )"

  def lower(self, source):
    return "(" + str(self._op) + "(" + str(self._lhs.lower(source)) + ") )"


class LetVertexExpr(VertexExpression):
  def __init__(self, expr, f):
    VertexExpression.__init__(self, f(expr).getType())
    self._expr = expr
    self._f = f

  def __str__(self):
    var_name = "x" + str(hash(self))
    return "(let " + var_name + "=" + str(self._expr) + " in " + str(
        self._f(PlaceholderVertexExpr(var_name))) + ")"

  def lower(self, source):
    var_name = "var" + str(source.GetNextName())
    return "[&, " + var_name + "=" + self._expr.lower(
        source) + "]{return " + self._f(
            PlaceholderVertexExpr(var_name)).lower(source) + ";}()"


def let(expr, f):
  return LetVertexExpr(expr, f)


class Conditional(VertexExpression):
  def __init__(self, pred, a, b):
    VertexExpression.__init__(self, a.getType())
    self._pred = _pred
    self._a = a
    self._b = b

  def __str__(self):
    return "(if (" + str(self._pred) + ") then (" + str(
        self.__a) + ") else (" + str(self._b) + "))"

  def lower(self, source):
    return "((" + self._pred.lower(source) + ") ? (" + self._a.lower(
        source) + ") : (" + self._b.lower(source) + "))"


def condition(pred, a, b):
  return Conditional(pred, a, b)
