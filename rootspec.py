#!/usr/bin/env python

# Copyright 2017 DIANA-HEP
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

import ast
import re
import struct
import numpy
import yaml

from zlib import decompress as zlib_decompress
try:
    from lzma import decompress as lzma_decompress
except ImportError:
    from backports.lzma import decompress as lzma_decompress
from lz4.block import decompress as lz4_decompress

class RootSpecError(Exception): pass

class FixedWidth(struct.Struct):
    def read(self, file, index):
        return self.unpack(file[index:index + self.size])[0]

class PascalString(object):
    readlittle = struct.Struct("B")
    readbig = struct.Struct(">I")

    def read(self, file, index):
        size, = self.readlittle(file[index:index + 1])
        start = index + 1
        if size == 255:
            size, = self.readbig(file[index + 1:index + 5])
            start = index + 5
        return file[start:start + size].tostring()

    def size(self, file, index):
        size, = self.readlittle(file[index:index + 1])
        if size < 255:
            return size + 1
        else:
            size, = self.readbig(file[index + 1:index + 5])
            return size + 5

class CString(object):
    def read(self, file, index):
        end = index
        while file[end] != 0:
            end += 1
        return self.data[index:end].tostring()

    def size(self, file, index):
        end = index
        while file[end] != 0:
            end += 1
        return end + 1 - index

readers = {
    "bool": FixedWidth("?"),
    "int8": FixedWidth("b"),
    "uint8": FixedWidth("B"),
    "int16": FixedWidth(">h"),
    "uint16": FixedWidth(">H"),
    "int32": FixedWidth(">i"),
    "uint32": FixedWidth(">I"),
    "int64": FixedWidth(">q"),
    "uint64": FixedWidth(">Q"),
    "float32": FixedWidth(">f"),
    "float64": FixedWidth(">d"),
    "string": PascalString,
    "cstring": CString}

def reader(format):
    if isinstance(format, dict) and len(format) == 1 and list(format.keys())[0] == "string":
        bytes = list(format.values())[0]
        assert isinstance(bytes, int)
        return FixedWidth(repr(bytes) + "s")
    elif format in readers:
        return readers[format]
    else:
        raise RootSpecError("unrecognized format: {0}".format(format))

def predicate(expr):
    def prependself(expr):
        if isinstance(expr, ast.Name):
            return ast.Attribute(ast.Name("self", ast.Load()), expr.id, ast.Load())
        elif isinstance(expr, ast.AST):
            for field in expr._field:
                setattr(expr, field, prependself(getattr(expr, field)))
            return expr
        elif isinstance(expr, list):
            return [prependself(x) for x in expr]
        else:
            return expr
    return prependself(ast.parse(expr).body[0].value)

class Where(object):
    def __init__(self, basenum, offset):
        self.basenum = basenum
        self.offset = offset
    
def propfunction(name, conditions):
    return compile(ast.parse("""
def {0}(self):
    return reader.read(self._file, self._base{1} + {2})
""".format(name, conditions.basenum, conditions.offset)), "<auto>", "exec")

class ObjectInFile(object):
    def __init__(self, file, base0):
        self._file = file
        self._base0 = base0

def declare(spec, conditions):
    if isinstance(spec, list):
        properties = {}
        for s in spec:
            properties.update(declare(s, conditions))
        return properties

    elif isinstance(spec, dict) and len(spec) == 1:
        (name, s), = spec.items()
        r = reader(s)
        variables = {"reader": r}
        exec(propfunction(name, conditions), variables)
        conditions.offset += r.size
        return {name: property(variables[name])}

    else:
        raise Exception

def declareclass(name, specification):
    return type(name, (ObjectInFile,), declare(specification[name]["properties"], Where(0, 0)))

specification = yaml.load(open("specification.yaml"))

TFile = declareclass("TFile", specification)

file = numpy.memmap("/home/pivarski/storage/data/TrackResonanceNtuple_uncompressed.root", dtype=numpy.uint8, mode="r")

tfile = TFile(file, 0)

print tfile.magic
print tfile.version
print tfile.begin
