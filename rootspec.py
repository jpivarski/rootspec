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
from collections import OrderedDict
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

    def __call__(self, file, index, parent=None):
        return self.unpack(file[index:index + self.size])[0]
    
class PascalString(object):
    def __repr__(self):
        return "PascalString()"

    readlittle = struct.Struct("B")
    readbig = struct.Struct(">I")

    def __call__(self, file, index, parent=None):
        size, = self.readlittle.unpack(file[index:index + 1])
        start = index + 1
        if size == 255:
            size, = self.readbig.unpack(file[index + 1:index + 5])
            start = index + 5
        return file[start:start + size].tostring()

    @classmethod
    def _sizeof(cls, file, index):
        size, = PascalString.readlittle.unpack(file[index:index + 1])
        if size < 255:
            return size + 1
        else:
            size, = PascalString.readbig.unpack(file[index + 1:index + 5])
            return size + 5

class CString(object):
    def __repr__(self):
        return "CString()"

    def __call__(self, file, index, parent=None):
        end = index
        while file[end] != 0:
            end += 1
        return self.data[index:end].tostring()

    @classmethod
    def _sizeof(cls, file, index):
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
    "string": PascalString(),
    "cstring": CString()}

Where = namedtuple("Where", ["base", "offset", "reader"])
Jumpto = namedtuple("Jumpto", ["expr", "reader"])
After = namedtuple("After", ["seek"])
AfterSize = namedtuple("After", ["seek", "type"])
Split = namedtuple("Split", ["predicates"])
Init = namedtuple("Init", ["base", "pos"])
Array = namedtuple("Array", ["size", "type"])

def interprettype(format):
    if isinstance(format, dict) and set(format.keys()) == set(["string"]):
        bytes = format["string"]
        assert isinstance(bytes, int)
        return FixedWidth(repr(bytes) + "s")

    elif isinstance(format, dict) and set(format.keys()) == set(["array", "size"]):
        return Array(format["size"], interprettype(format["array"]))

    elif format in readers:
        return readers[format]

    else:
        return format

def expandifs(spec, base, offset, inits):
    assert isinstance(spec, list)
    fields = OrderedDict()
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

                thisfields, thisafter, _ = expandifs(consequent, base, offset, inits)

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
            
            if itemindex + 1 < len(spec):
                inits.append(Init(base + 1, init))
                base += 1
                offset = 0
            else:
                lastinit = Init("END", init)

        else:
            (name, format), = item.items()
            assert name not in fields

            jumpto = None
            if isinstance(format, dict) and "type" in format:
                jumpto = format.get("at", None)
                format = format["type"]

            reader = interprettype(format)
            if jumpto is None:
                fields[name] = Where(base, offset, reader)
            else:
                fields[name] = Jumpto(jumpto, reader)

            if isinstance(reader, Array):
                if itemindex + 1 < len(spec):
                    inits.append(Init(base + 1, AfterSize("_base{0} + {1}".format(base, offset), reader)))
                    base += 1
                    offset = 0
                else:
                    lastinit = Init("END", AfterSize("_base{0} + {1}".format(base, offset), reader))

            elif hasattr(reader, "size"):
                offset += reader.size
                if itemindex + 1 == len(spec):
                    lastinit = Init("END", After("_base{0} + {1}".format(base, offset)))
                
            else:
                if itemindex + 1 < len(spec):
                    inits.append(Init(base + 1, AfterSize("_base{0} + {1}".format(base, offset), reader)))
                    base += 1
                    offset = 0
                else:
                    lastinit = Init("END", AfterSize("_base{0} + {1}".format(base, offset), reader))

    return fields, After("_base{0} + {1}".format(base, offset)), lastinit

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

def pythonexpr(expr):
    return prependself(ast.parse(expr).body[0].value)

def addtoreaders(reader, readers):
    if isinstance(reader, str):
        return reader
    else:
        found = False
        for rn, r in readers.items():
            if r is reader:
                return rn
        rn = "reader{0}".format(len(readers))
        readers[rn] = reader
        return rn

def pythoninit(inits):
    readers = {}
    def setbase(init):
        if isinstance(init.pos, AfterSize):
            if isinstance(init.pos.type, Array) and isinstance(init.pos.type.type, FixedWidth):
                rn = addtoreaders(init.pos.type.type, readers)
                out = ast.parse("self._base{0} = REPLACEME + {1}.size * REPLACEME".format(init.base, rn)).body
                out[0].value.left = pythonexpr(init.pos.seek)
                out[0].value.right.right = pythonexpr(init.pos.type.size)
                return out

            elif isinstance(init.pos.type, Array):
                rn = addtoreaders(init.pos.type.type, readers)
                out = ast.parse("""
self._base{0} = REPLACEME
i = 0
while i < REPLACEME:
    print i, self._base{0}
    self._base{0} += {1}._sizeof(self._file, self._base{0})
    i += 1
""".format(init.base, rn)).body
                out[0].value = pythonexpr(init.pos.seek)
                out[2].test.comparators[0] = pythonexpr(init.pos.type.size)
                return out

            else:
                rn = addtoreaders(init.pos.type, readers)
                out = ast.parse("self._base{0} = REPLACEME + {1}._sizeof(self._file, REPLACEME)".format(init.base, rn)).body
                out[0].value.left = pythonexpr(init.pos.seek)
                out[0].value.right.args[1] = pythonexpr(init.pos.seek)
                return out

        elif isinstance(init.pos, After):
            out = ast.parse("self._base{0} = REPLACEME".format(init.base)).body
            out[0].value = pythonexpr(init.pos.seek)
            return out

        else:
            out = None
            for predicate, consequent in reversed(init.pos):
                if predicate is None:
                    assert out is None
                    out = setbase(Init(init.base, consequent))
                else:
                    tmp = ast.parse("if REPLACEME:\n  REPLACEME\nelse:  REPLACEME").body
                    tmp[0].test = pythonexpr(predicate)
                    tmp[0].body = setbase(Init(init.base, consequent))
                    tmp[0].orelse = out
                    out = tmp
            return out

    out = ast.parse("def __init__(self, file, base0, parent=None):\n  Cursor.__init__(self, file, base0, parent)")
    for x in inits:
        tmp = setbase(x)
        import meta
        print meta.dump_python_source(ast.Module(tmp))

        out.body[0].body.extend(tmp)
    return out, readers

def pythonprop(prop):
    readers = {}
    def recurse(prop):
        if isinstance(prop, (Where, Jumpto)):
            if isinstance(prop, Where):
                start = ast.parse("offset = self._base{0} + {1}".format(prop.base, prop.offset)).body
            elif isinstance(prop, Jumpto):
                start = ast.parse("offset = REPLACEME").body
                start[0].value = pythonexpr(prop.expr)
            else:
                raise AssertionError

            if isinstance(prop.reader, Array):
                rn = addtoreaders(prop.reader.type, readers)
                sizer = ast.parse("size = REPLACEME").body
                sizer[0].value = pythonexpr(prop.reader.size)
                filler = ast.parse("""
out = [None] * size
i = 0
while i < size:
    out[i] = {0}(self._file, offset, self)
    offset = out[i]._baseEND
    i += 1
return out
""".format(rn)).body
                return start + sizer + filler

            else:
                rn = addtoreaders(prop.reader, readers)
                return start + ast.parse("return {0}(self._file, offset, self)".format(rn)).body

        elif isinstance(prop, Split):
            out = None
            for predicate, consequent in reversed(prop.predicates):
                if predicate is None:
                    assert out is None
                    out = recurse(consequent)
                else:
                    tmp = ast.parse("if REPLACEME:\n  REPLACEME\nelse:\n  REPLACEME").body
                    tmp[0].test = pythonexpr(predicate)
                    tmp[0].body = recurse(consequent)
                    tmp[0].orelse = out
                    out = tmp
            return out

    out = ast.parse("def PROPERTY(self):\n  REPLACEME")
    out.body[0].body = recurse(prop)
    return out, readers

class Cursor(object):
    def __init__(self, file, base0, parent=None):
        self._file = file
        self._base0 = base0
        self._parent = parent
        print "start", self.__class__, "at", base0

    def __repr__(self):
        return "<{0} in {1} at {2}>".format(self.__class__.__name__, repr(self._file.filename), self._base0)

    def base(self):
        return self._base0

    @classmethod
    def _sizeof(cls, file, pos):
        print "_sizeof", cls, pos
        obj = cls(file, pos)
        return obj._baseEND - obj._base0

    @classmethod
    def _debug(cls, method=None):
        import meta

        def recurse(x, readers):
            if isinstance(x, ast.Name) and x.id in readers:
                return ast.Name(repr(readers[x.id]), ast.Load())
            elif isinstance(x, ast.AST):
                for field in x._fields:
                    setattr(x, field, recurse(getattr(x, field), readers))
                return x
            elif isinstance(x, list):
                return [recurse(y, readers) for y in x]
            else:
                return x

        if method is not None:
            source, readers = cls._source[method]
            recurse(source, readers)
            source.body[0].name = "{0}.{1}".format(cls.__name__, method)
            return meta.dump_python_source(source)

        else:
            out = "class {0}(Cursor):".format(cls.__name__)
            for method, (source, readers) in cls._source.items():
                recurse(source, readers)
                source.body[0].name = method
                if method != "__init__":
                    out += "\n    @property"
                out += meta.dump_python_source(source).replace("\n", "\n    ").rstrip() + "\n"
            return out

def declareclass(classname, spec):
    inits = []
    fields, after, lastinit = expandifs(spec["properties"], 0, 0, inits)
    inits.append(lastinit)

    out = type(classname, (Cursor,), {})
    out._source = OrderedDict([("__init__", pythoninit(inits))])
    for name, prop in fields.items():
        out._source[name] = pythonprop(prop)
    return out

def declare(specification):
    classes = {}
    for name, spec in specification.items():
        classes[name] = declareclass(name, spec)

    for cls in classes.values():
        for name, (source, readers) in cls._source.items():
            env = {"Cursor": Cursor}
            env.update(classes)
            env.update(readers)
            exec(compile(source, "<auto>", "exec"), env)

            if name == "__init__":
                setattr(cls, name, MethodType(env["__init__"], None, cls))
            else:
                setattr(cls, name, property(env["PROPERTY"]))

    return classes

classes = declare(yaml.load(open("specification.yaml")))

file = numpy.memmap("/home/pivarski/storage/data/TrackResonanceNtuple_uncompressed.root", dtype=numpy.uint8, mode="r")

print classes["TKeys"]._debug()

tfile = classes["TFile"](file, 0)
print "tfile.magic", repr(tfile.magic)
print "tfile.version", tfile.version
print "tfile.begin", tfile.begin
print "tfile.end", tfile.end
print "tfile.seekfree", tfile.seekfree
print "tfile.nbytesfree", tfile.nbytesfree
print "tfile.nfree", tfile.nfree
print "tfile.nbytesname", tfile.nbytesname
print "tfile.units", tfile.units
print "tfile.compression", tfile.compression
print "tfile.seekinfo", tfile.seekinfo
print "tfile.nbytesinfo", tfile.nbytesinfo
print "tfile.uuid", repr(tfile.uuid)

tdirectory = tfile.dir
print "tdirectory.version", tdirectory.version
print "tdirectory.ctime", tdirectory.ctime
print "tdirectory.mtime", tdirectory.mtime
print "tdirectory.nbyteskeys", tdirectory.nbyteskeys
print "tdirectory.nbytesname", tdirectory.nbytesname
print "tdirectory.seekdir", tdirectory.seekdir
print "tdirectory.seekparent", tdirectory.seekparent
print "tdirectory.seekkeys", tdirectory.seekkeys

keys = tdirectory.keys
header = keys.header
print "header.bytes", header.bytes
print "header.version", header.version
print "header.objlen", header.objlen
print "header.datetime", header.datetime
print "header.keylen", header.keylen
print "header.cycle", header.cycle
print "header.seekkey", header.seekkey
print "header.seekpdir", header.seekpdir
print "header.classname", repr(header.classname)
print "header.name", repr(header.name)
print "header.title", repr(header.title)

print "keys.nkeys", keys.nkeys
# print "keys.keys", keys.keys
# >>> classes["TKeys"](file, 12069632)
# 0 12069708
