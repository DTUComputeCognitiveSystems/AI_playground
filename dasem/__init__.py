"""dasem."""

from __future__ import absolute_import

from .fullmonty import Word2Vec

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__all__ = ('Word2Vec',)
