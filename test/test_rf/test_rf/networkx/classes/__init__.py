"""
=======
Classes
=======

The :mod:`networkx.classes` module contains the graph classes.
"""

from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.reportviews import *
from networkx.classes.coreviews import *
from networkx.classes.graphviews import *
from networkx.classes.function import *
from networkx.classes.filters import *

__all__ = [
    "Graph",
    "DiGraph",
    "MultiGraph",
    "MultiDiGraph",
] + reportviews.__all__ + coreviews.__all__ + graphviews.__all__ + function.__all__ + filters.__all__
