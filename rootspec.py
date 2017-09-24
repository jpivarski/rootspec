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

class Instance(object):
    def __init__(self, file, *pos):
        self.file = file
        self._pos = pos

def bigendian(format):
    if format.startswith(">") or format.startswith("<") or format.startswith("!") or format.startswith("="):
        return format
    else:
        return ">" + format

def declaresimple(posindex, offset, pspecdata):
    format = bigendian(pspecdata)
    size = struct.calcsize(format)
    def prop(self):
        start = self._pos[posindex] + offset
        end = start + size
        print "start", start, "end", end, "data", " ".join("%02x" % x for x in self.file[start:end]), "format", format, "property", struct.unpack(format, self.file[start:end])
        return struct.unpack(format, self.file[start:end])[0]
    return property(prop), size

def declare(spec):
    properties = {}
    posindex = 0
    offset = 0
    for pspec in spec["properties"]:
        assert isinstance(pspec, dict) and len(pspec) == 1
        (name, pspecdata), = pspec.items()

        if isinstance(pspecdata, str):
            properties[name], size = declaresimple(posindex, offset, pspecdata)
        else:
            break

        offset += size

    return properties

specification = yaml.load(open("specification.yaml"))

TFile = type("TFile", (Instance,), declare(specification["TFile"]))

file = numpy.memmap("/home/pivarski/storage/data/TrackResonanceNtuple_uncompressed.root", dtype=numpy.uint8, mode="r")

tfile = TFile(file, 0)

print tfile.magic
print tfile.version
print tfile.begin
