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
from collections import namedtuple
from types import MethodType

from zlib import decompress as zlib_decompress
try:
    from lzma import decompress as lzma_decompress
except ImportError:
    from backports.lzma import decompress as lzma_decompress
from lz4.block import decompress as lz4_decompress

class RootSpecError(Exception): pass

class FixedWidth(struct.Struct):
    def __repr__(self):
        return "FixedWidth({0})".format(repr(self.format))

    def __call__(self, file, index):
        return self.unpack(file[index:index + self.size])[0]

class PascalString(object):
    def __repr__(self):
        return "PascalString()"

    readlittle = struct.Struct("B")
    readbig = struct.Struct(">I")

    def __call__(self, file, index):
        size, = self.readlittle(file[index:index + 1])
        start = index + 1
        if size == 255:
            size, = self.readbig(file[index + 1:index + 5])
            start = index + 5
        return file[start:start + size].tostring()

    # def size(self, file, index):
    #     size, = self.readlittle(file[index:index + 1])
    #     if size < 255:
    #         return size + 1
    #     else:
    #         size, = self.readbig(file[index + 1:index + 5])
    #         return size + 5

class CString(object):
    def __repr__(self):
        return "CString()"

    def __call__(self, file, index):
        end = index
        while file[end] != 0:
            end += 1
        return self.data[index:end].tostring()

    # def size(self, file, index):
    #     end = index
    #     while file[end] != 0:
    #         end += 1
    #     return end + 1 - index

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

Where = namedtuple("Where", ["base", "offset", "reader"])
Jumpto = namedtuple("Jumpto", ["expr", "reader"])
After = namedtuple("After", ["base", "end"])
AfterSize = namedtuple("After", ["base", "end", "type"])
Split = namedtuple("Split", ["predicates"])
Init = namedtuple("Init", ["base", "pos"])

def reader(format):
    if isinstance(format, dict) and len(format) == 1 and list(format.keys())[0] == "string":
        bytes = list(format.values())[0]
        assert isinstance(bytes, int)
        return FixedWidth(repr(bytes) + "s")
    elif format in readers:
        return readers[format]
    else:
        return format

def expandifs(spec, base, offset, inits):
    assert isinstance(spec, list)
    fields = {}
    for itemindex, item in enumerate(spec):
        if set(item.keys()) == set(["if"]):
            assert isinstance(item["if"], list)

            predicates = []
            names = set()
            init = []
            for case in item["if"]:
                if set(case.keys()) == set(["case", "then"]):
                    predicate = case["case"]
                    consequent = case["then"]
                elif set(case.keys()) == set(["else"]):
                    predicate = None
                    consequent = case["else"]
                else:
                    raise AssertionError

                thisfields, thisafter = expandifs(consequent, base, offset, inits)

                for name in thisfields:
                    if name not in fields:
                        fields[name] = Split([(p, None) for p in predicates])
                    fields[name].predicates.append((predicate, thisfields[name]))

                for name in names:
                    if name not in thisfields:
                        fields[name].predicates.append((predicate, None))

                predicates.append(predicate)
                names.update(thisfields)
                init.append((predicate, thisafter))
            
            base += 1
            offset = 0
            inits.append(Init(base, init))

        else:
            (name, format), = item.items()
            assert name not in fields

            jumpto = None
            if isinstance(format, dict) and "type" in format:
                jumpto = format.get("at", None)
                format = format["type"]

            r = reader(format)
            if jumpto is None:
                fields[name] = Where(base, offset, r)
            else:
                fields[name] = Jumpto(jumpto, r)

            if hasattr(r, "size"):
                offset += r.size
            else:
                inits.append(Init(base, AfterSize(base, offset, r)))
                base += 1
                offset = 0

    return fields, After(base, offset)

def prependself(expr):
    if isinstance(expr, ast.Name):
        return ast.parse("self.{0}".format(expr.id)).body[0].value
    elif isinstance(expr, ast.AST):
        for field in expr._fields:
            setattr(expr, field, prependself(getattr(expr, field)))
        return expr
    elif isinstance(expr, list):
        return [prependself(x) for x in expr]
    else:
        return expr

def pythonpredicate(expr):
    return prependself(ast.parse(expr).body[0].value)

def pythoninit(inits):
    def setbase(init):
        if isinstance(init.pos, AfterSize):
            return ast.parse("self._base{0} = {1}._sizeof(self._file, self._base{2} + {3})".format(init.base, init.pos.type, init.pos.base, init.pos.end)).body[0]

        else:
            out = None
            for predicate, consequent in reversed(init.pos):
                if predicate is None:
                    assert out is None
                    out = ast.parse("self._base{0} = self._base{1} + {2}".format(init.base, consequent.base, consequent.end)).body[0]
                else:
                    tmp = ast.parse("if REPLACEME:\n  self._base{0} = self._base{1} + {2}\nelse:  REPLACEME".format(init.base, consequent.base, consequent.end)).body[0]
                    tmp.test = pythonpredicate(predicate)
                    tmp.orelse = [out]
                    out = tmp
            return out

    out = ast.parse("def __init__(self, file, base0):\n  Cursor.__init__(self, file, base0)")
    out.body[0].body.extend([setbase(x) for x in inits])
    return out

def pythonprop(prop):
    readers = {}
    def recurse(prop):
        if isinstance(prop, (Where, Jumpto)):
            if isinstance(prop.reader, str):
                rn = prop.reader
            else:
                found = False
                for rn, r in readers.items():
                    if r is prop.reader:
                        found = True
                        break
                if not found:
                    rn = "reader{0}".format(len(readers))
                    readers[rn] = prop.reader

            if isinstance(prop, Where):
                return ast.parse("return {0}(self._file, self._base{1} + {2})".format(rn, prop.base, prop.offset)).body[0]
            elif isinstance(prop, Jumpto):
                out = ast.parse("return {0}(self._file, REPLACEME)".format(rn)).body[0]
                out.value.args[1] = prependself(ast.parse(prop.expr).body[0].value)
                return out

        elif isinstance(prop, Split):
            out = None
            for predicate, consequent in reversed(prop.predicates):
                if predicate is None:
                    assert out is None
                    out = recurse(consequent)
                else:
                    tmp = ast.parse("if REPLACEME:\n  REPLACEME\nelse:\n  REPLACEME").body[0]
                    tmp.test = pythonpredicate(predicate)
                    tmp.body = [recurse(consequent)]
                    tmp.orelse = [out]
                    out = tmp
            return out

    out = ast.parse("def PROPERTY(self):\n  REPLACEME")
    out.body[0].body = [recurse(prop)]
    return out, readers

class Cursor(object):
    def __init__(self, file, base0):
        self._file = file
        self._base0 = base0

    def __repr__(self):
        return "<{0} in {1} at {2}>".format(self.__class__.__name__, repr(self._file.filename), self._base0)

    @classmethod
    def _sizeof(cls, file, pos):
        return 0

def declareclass(classname, spec):
    inits = []
    fields, after = expandifs(spec["properties"], 0, 0, inits)

    out = type(classname, (Cursor,), {})
    out.__init = pythoninit(inits)
    out.__properties = {}
    for name, prop in fields.items():
        out.__properties[name] = pythonprop(prop)
    return out

def declare(specification):
    classes = {}
    for name, spec in specification.items():
        classes[name] = declareclass(name, spec)

    for cls in classes.values():
        env = classes.copy()
        env["Cursor"] = Cursor
        exec(compile(cls.__init, "<auto>", "exec"), env)
        cls.__init__ = MethodType(env["__init__"], None, cls)

        for name, (source, readers) in cls.__properties.items():
            env = classes.copy()
            env.update(readers)

            exec(compile(source, "<auto>", "exec"), env)
            setattr(cls, name, property(env["PROPERTY"]))

    return classes

classes = declare(yaml.load(open("specification.yaml")))

file = numpy.memmap("/home/pivarski/storage/data/TrackResonanceNtuple_uncompressed.root", dtype=numpy.uint8, mode="r")
tfile = classes["TFile"](file, 0)
print "magic", tfile.magic
print "version", tfile.version
print "begin", tfile.begin
print "end", tfile.end
print "seekfree", tfile.seekfree
print "nbytesfree", tfile.nbytesfree
print "nfree", tfile.nfree
print "nbytesname", tfile.nbytesname
print "units", tfile.units
print "compression", tfile.compression
print "seekinfo", tfile.seekinfo
print "nbytesinfo", tfile.nbytesinfo
print "uuid", repr(tfile.uuid)
print "dir.version", tfile.dir.version
print "dir.ctime", tfile.dir.ctime
print "dir.mtime", tfile.dir.mtime
print "dir.nbyteskeys", tfile.dir.nbyteskeys
print "dir.nbytesname", tfile.dir.nbytesname
print "dir.seekdir", tfile.dir.seekdir
print "dir.seekparent", tfile.dir.seekparent
print "dir.seekkeys", tfile.dir.seekkeys
