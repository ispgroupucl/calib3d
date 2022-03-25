from mako.template import Template
import numpy as np

from calib3d import Calib

class CudaCalib(Calib):
    datatypes = {
            'width':  ('__align__(8) int',    '',     8),
            'height': ('__align__(8) int',    '',     8),
            'T':      ('__align__(8) double', '[3]',  24),
            'K':      ('__align__(8) double', '[9]',  72),
            'kc':     ('__align__(8) double', '[5]',  40),
            'R':      ('__align__(8) double', '[9]' , 72),
            'C':      ('__align__(8) double', '[3]' , 24),
            'P':      ('__align__(8) double', '[12]', 96),
            'Pinv':   ('__align__(8) double', '[12]', 96),
            'Kinv':   ('__align__(8) double', '[9]',  72),
        }

    @classmethod
    def struct_str(cls):
        return Template("""
            struct calib_t
            {
                %for var in data:
                ${var[1]} ${var[0]}${var[2]};
                %endfor
            };
        """).render(data=[(name, dtype, size) for name, (dtype, size, _) in  cls.datatypes.items()])

    @classmethod
    def memsize(cls):
        return sum([size for attr, (_, _, size) in cls.datatypes.items()])

    @classmethod
    def from_calib(cls, calib):
        new_calib = cls.__new__(cls)
        new_calib.update(**calib.dict)
        return new_calib

    def memset(self, ptr, callback):
        # check function 'pycuda.tools.dtype_to_ctype'
        offset = 0
        for name, (_, _, size) in self.datatypes.items():
            value = self.__dict__[name]
            if isinstance(value, int):
                data = memoryview(np.int32(value))
            elif isinstance(value, float):
                data = memoryview(np.float32(value))
            else:
                data = memoryview(value)
            callback(int(ptr)+offset, data)
            offset += size
