
"""
setup.py
"""

from distutils.core import setup, Extension


stringSegment_module = Extension('_stringSegment',
#                           swig_opts = ['-c++'],
#                            include_dirs = ['./.']
#                           library_dirs=[,],
#                           libraries=['usr/local/lib',],
                           sources=['stringSegment_wrap.cxx', 'stringSegment.cpp',
                           'pre_process.cpp'],
                           depends = ['stringSegment.h','pre_process.hpp'],
                           extra_compile_args=['-std=c++11',]
                           )

setup (name = 'stringSegment',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [stringSegment_module],
       py_modules = ["stringSegment"],
       )
