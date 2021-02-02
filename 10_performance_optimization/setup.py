from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("euclid.pyx", annotate=True, compiler_directives={'language_level': "3"})
)
