"""
NetworkX
========

NetworkX is a Python package for the creation, manipulation, and study of the
structure, dynamics, and functions of complex networks.

See https://networkx.org for complete documentation.
"""

__version__ = "3.6.1"


from networkx import utils

config = utils.backends._set_configs_from_environment()
utils.config = utils.configs.config = config

_dispatchable = utils.backends._dispatchable


from networkx.exception import NetworkXError, NetworkXException, NetworkXNotImplemented
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph
from networkx import convert
from networkx.utils.misc import _clear_cache
from networkx.drawing.layout import spring_layout, circular_layout, random_layout, shell_layout, spectral_layout
from networkx.drawing.nx_pylab import draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, draw_networkx_edge_labels
from networkx.convert_matrix import to_numpy_array
from networkx.classes.function import selfloop_edges, nodes, edges


def __getattr__(name):
    if name == "random_tree":
        raise AttributeError(
            "nx.random_tree was removed in version 3.4. Use `nx.random_labeled_tree` instead.\n"
            "See: https://networkx.org/documentation/latest/release/release_3.4.html"
        )
    raise AttributeError(f"module 'networkx' has no attribute '{name}'")
