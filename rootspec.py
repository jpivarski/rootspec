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
import string
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
    def __init__(self, format, dtype):
        super(FixedWidth, self).__init__(format)
        self.dtype = dtype

    def __repr__(self):
        return "FixedWidth({0}, {1})".format(repr(self.format), repr(self.dtype))

    def __call__(self, file, index, parent=None):
        return self.unpack(file[index:index + self.size])[0]

    def _sizeof(self, file, index):
        return self.size

    def array(self, file, index, size):
        return file[index:index + size*self.dtype.itemsize].view(self.dtype)

class Postprocess(object):
    def __init__(self, reader, name, expr):
        self.reader = reader
        self.name = name
        self.expr = expr
        env = {}
        for x in dir(numpy):
            env[x] = getattr(numpy, x)
        exec(compile(ast.parse("def postprocess({0}):\n  return {1}".format(name, expr)), "<auto>", "exec"), env)
        self.fcn = env["postprocess"]

    def __repr__(self):
        return "Postprocess({0}, {1}, {2})".format(self.reader, repr(self.name), repr(self.expr))

    def __call__(self, file, index, parent=None):
        return self.fcn(self.reader(file, index, parent))

    def _sizeof(self, file, index):
        return self.reader._sizeof(file, index)

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
    "bool": FixedWidth("?", numpy.dtype(numpy.bool_)),
    "int8": FixedWidth("b", numpy.dtype("i1")),
    "uint8": FixedWidth("B", numpy.dtype("u1")),
    "int16": FixedWidth(">h", numpy.dtype(">i2")),
    "uint16": FixedWidth(">H", numpy.dtype(">u2")),
    "int32": FixedWidth(">i", numpy.dtype(">i4")),
    "uint32": FixedWidth(">I", numpy.dtype(">u4")),
    "int64": FixedWidth(">q", numpy.dtype(">i8")),
    "uint64": FixedWidth(">Q", numpy.dtype(">u8")),
    "float32": FixedWidth(">f", numpy.dtype(">f4")),
    "float64": FixedWidth(">d", numpy.dtype(">f8")),
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
        return FixedWidth(repr(bytes) + "s", None)

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
            assert isinstance(item["if"], list) and len(item["if"]) > 0

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

            if init[-1][0] is not None:
                init.append((None, After("_base{0} + {1}".format(base, offset))))

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
            postprocess = None
            if isinstance(format, dict) and "type" in format:
                jumpto = format.get("at", None)
                postprocess = format.get("postprocess", None)
                format = format["type"]

            reader = interprettype(format)
            if postprocess is not None:
                reader = Postprocess(reader, name, postprocess)

            if jumpto is None:
                fields[name] = Where(base, offset, reader)
                initseek = "_base{0} + {1}".format(base, offset)
            else:
                fields[name] = Jumpto(jumpto, reader)
                initseek = jumpto

            if isinstance(reader, Array):
                if itemindex + 1 < len(spec):
                    inits.append(Init(base + 1, AfterSize(initseek, reader)))
                    base += 1
                    offset = 0
                else:
                    lastinit = Init("END", AfterSize(initseek, reader))

            elif hasattr(reader, "size"):
                offset += reader.size
                if itemindex + 1 == len(spec):
                    if jumpto is None:
                        initseek = "_base{0} + {1}".format(base, offset)
                    lastinit = Init("END", After(initseek))
                
            else:
                if itemindex + 1 < len(spec):
                    inits.append(Init(base + 1, AfterSize(initseek, reader)))
                    base += 1
                    offset = 0
                else:
                    lastinit = Init("END", AfterSize(initseek, reader))

    return fields, After("_base{0} + {1}".format(base, offset)), lastinit

reservednumpy = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64"]
def prependself(expr):
    if isinstance(expr, ast.Name) and expr.id in reservednumpy:
        return ast.parse("numpy.{0}".format(expr.id)).body[0].value
    elif isinstance(expr, ast.Name):
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
    return prependself(ast.parse(re.sub(r"\$size\b", "(_baseEND - _base0)", re.sub(r"\$pos\b", "_base0", expr))).body[0].value)

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

def pythoninit(inits, asserts):
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
                    assert out is not None
                    tmp = ast.parse("if REPLACEME:\n  REPLACEME\nelse:  REPLACEME").body
                    tmp[0].test = pythonexpr(predicate)
                    tmp[0].body = setbase(Init(init.base, consequent))
                    tmp[0].orelse = out
                    out = tmp
            return out

    out = ast.parse("def __init__(self, file, base0, parent=None):\n  Cursor.__init__(self, file, base0, parent)")

    for x in inits:
        out.body[0].body.extend(setbase(x))

    assert isinstance(asserts, list)
    for x in asserts:
        tmp = ast.parse("assert REPLACEME, 'in {0} starting at index {1} expected {2}'.format(self.__class__.__name__, base0, 'REPLACEME')").body
        tmp[0].test = pythonexpr(x)
        tmp[0].msg.args[2].s = x
        out.body[0].body.extend(tmp)

    return out, readers

def pythonprop(classname, name, prop):
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

                if isinstance(prop.reader.type, FixedWidth):
                    filler = ast.parse("return {0}.array(self._file, offset, size)".format(rn)).body
                else:
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
                    if out is None:
                        out = ast.parse("raise ValueError('property {0}.{1} has no value under this set of conditions')".format(classname, name)).body
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

    def __repr__(self):
        return "<{0} in {1} at {2}>".format(self.__class__.__name__, repr(self._file.filename), self._base0)

    def hexdump(self, at=None, size=160, offset=0, format="%02x"):
        if at is None:
            at = self._base0
        elif isinstance(at, Cursor):
            at = at._base0
        pos = at + offset

        out = []
        for linepos in range(pos, pos + size, 16):
            data = self._file[linepos:min(linepos + 16, pos + size)]
            line = [format % x for x in data]
            text = [chr(x) if chr(x) in string.printable[:-5] else "." for x in data]
            if len(line) < 16:
                diff = 16 - len(line)
                line.extend(["  "] * diff)
                text.extend([" "] * diff)
            out.append("{0:08d}  {1}  {2}  |{3}|".format(linepos, " ".join(line[:8]), " ".join(line[8:]), "".join(text)))
        return "\n".join(out)

    @classmethod
    def _sizeof(cls, file, pos):
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
    out._source = OrderedDict([("__init__", pythoninit(inits, spec.get("assert", [])))])
    for name, prop in fields.items():
        out._source[name] = pythonprop(classname, name, prop)
    return out

def declare(specification):
    classes = {}
    for name, spec in specification.items():
        classes[name] = declareclass(name, spec)

    for cls in classes.values():
        for name, (source, readers) in cls._source.items():
            env = {"Cursor": Cursor, "numpy": numpy}
            env.update(classes)
            env.update(readers)
            exec(compile(source, "<auto>", "exec"), env)

            if name == "__init__":
                setattr(cls, name, MethodType(env["__init__"], None, cls))
            else:
                setattr(cls, name, property(env["PROPERTY"]))

    return classes

classes = declare(yaml.load(open("specification.yaml")))

file = numpy.memmap("histograms.root", dtype=numpy.uint8, mode="r")

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

print "keys.nkeys", keys.nkeys, len(keys.keys)

for key in keys.keys:
    print "key.bytes", key.bytes
    print "key.version", key.version
    print "key.objlen", key.objlen
    print "key.datetime", key.datetime
    print "key.keylen", key.keylen
    print "key.cycle", key.cycle
    print "key.seekkey", key.seekkey
    print "key.seekpdir", key.seekpdir
    print "key.classname", repr(key.classname)
    print "key.name", repr(key.name)
    print "key.title", repr(key.title)

# >>> classes["TKeys"](file, 12069632)
# 0 12069708

key = keys.keys[1]

print key.hexdump(key.seekkey + key.keylen, format="%3d")

hist = classes["TH1F"](key._file, key.seekkey + key.keylen)

print repr(hist.named.name), repr(hist.named.title)

print hist.ncells

print hist.versionheader.bytecount
print hist.named.versionheader.bytecount
print hist.attline.versionheader.bytecount
print hist.attfill.versionheader.bytecount
print hist.attmarker.versionheader.bytecount

print repr(hist.xaxis.named.name), repr(hist.xaxis.named.title), hist.xaxis.nbins, hist.xaxis.xmin, hist.xaxis.xmax, hist.xaxis.binedges, hist.xaxis.first, hist.xaxis.last, hist.xaxis.bits2, hist.xaxis.time, repr(hist.xaxis.tfmt)

print repr(hist.yaxis.named.name), repr(hist.yaxis.named.title), hist.yaxis.nbins, hist.yaxis.xmin, hist.yaxis.xmax, hist.yaxis.binedges, hist.yaxis.first, hist.yaxis.last, hist.yaxis.bits2, hist.yaxis.time, repr(hist.yaxis.tfmt)

print repr(hist.zaxis.named.name), repr(hist.zaxis.named.title), hist.zaxis.nbins, hist.zaxis.xmin, hist.zaxis.xmax, hist.zaxis.binedges, hist.zaxis.first, hist.zaxis.last, hist.zaxis.bits2, hist.zaxis.time, repr(hist.zaxis.tfmt)

print hist.versionheader.version
print hist.entries
print hist.tsumw, hist.tsumw2, hist.tsumwx, hist.tsumwx2
print hist.max, hist.min, hist.norm, hist.contour_size, hist.contour
print hist.sumw2_size, hist.sumw2

print hist.buffer.tolist()
