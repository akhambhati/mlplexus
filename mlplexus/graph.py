"""
Implements the basic infrastructure of
Graph, GraphState.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/04/09
"""

import numpy as np

from mlplexus.checks import (checkArrDims, checkArrDTypeStr, checkArrLen,
                             checkArrSqr, checkNone, checkPath, checkType)
from mlplexus.exception import (mlPlexusException, mlPlexusIOError,
                                mlPlexusNotImplemented, mlPlexusTypeError,
                                mlPlexusValueError)
from mlplexus.helper import class_as_string


class GraphArch(object):
    """
    Class to define the immutable properties of the Graph Architecture.

    GraphArch is a collection of individual nodes, each with an
    immutable identity in the system.

    *NOTE*: Any mutable attributes of the Graph are identified in objects
    derived from the GraphState class, _not_ in the Graph class.
    """

    def __init__(self, node_id=None, **node_attr):
        """
        Initialize graph with nodes of specific ID and attributes.

        Parameters
        ----------
            node_id: np.ndarray(str,)
                List of strings, where each string identifies a node.

            node_attr: key/value -->
                         value=np.ndarray(obj,)
                Each attribute is defined as a key-value pair.
        """

        # First, handle the node_id
        if checkNone(node_id) or (not checkType(node_id, np.ndarray)):
            raise mlPlexusTypeError('Must supply a node_id of type np.ndarray')
        node_id = node_id.squeeze()
        if not checkArrDims(node_id, 1):
            raise mlPlexusException(
                'Must supply a node_id with a 1d-array of identifiers')
        if not checkArrDTypeStr(node_id):
            raise mlPlexusTypeError(
                'Must supply a node_id containing strings as identifiers')
        n_node = len(node_id)

        # Second, handle the incoming attributes
        if not checkType(node_attr, dict):
            raise mlPlexusTypeError('All attributes must be supplied in'
                                    ' conventional Python `dict` format')

        # Construct a list of numpy dtypes for the structure
        dtype_tuple = []
        for key in [*node_attr]:
            key_attr = node_attr[key].squeeze()
            if not checkType(key_attr, np.ndarray):
                raise mlPlexusTypeError('Must supply numpy array for attribute'
                                        ' \'{}\'.'.format(key))
            if not checkArrLen(key_attr, n_node):
                raise mlPlexusTypeError('Must supply numpy array with'
                                        ' same number of values as nodes'
                                        ' in first dimension, for attribute'
                                        ' \'{}\'.'.format(key))
            dtype_tuple.append((key, key_attr.dtype, key_attr.shape[1:]))
        dtype_tuple.append(('node_id', node_id.dtype, node_id.shape[1:]))

        # Generate a Structured Array from the attributes
        arr_node_attr = np.empty(n_node, dtype=dtype_tuple)
        arr_node_attr['node_id'] = node_id
        for key in [*node_attr]:
            arr_node_attr[key] = node_attr[key]

        # Set attributes for the class attributes
        self._define_graph(arr_node_attr)

    def _define_graph(self, node_attr):
        """Define class instance attributes from node attribute array"""
        self.node_attr = node_attr
        self.node_id = node_attr['node_id']
        self.n_node = len(node_attr)
        self.node_attr.flags.writeable = False

    def get_node(self, node):
        """Return the attributes for a node"""
        node_ix = self.node_attr['node_id'] == node
        return self.node_attr[node_ix]

    def __str__(self):
        """Prettified string representation of node ids."""
        s = '\n::NODES::\n'
        return s + '\n'.join(self.node_id)

    def save(self, graph_name, graph_dir):
        """
        Save the constructed Graph Architecture.

        The Graph Architecture can be fully reconstructed from the attribute,
        node_attr.

        Parameters
        ----------
            graph_name: str
                Unique name given to the defined Graph Architecture.
            graph_dir: str
                The directory in which the Graph Architecture will be saved.

        Returns
        -------
            graph_path: str
                Saved file will have the path,
                <graph_dir>/mlplexus.graph.GraphArch.<graph_name>.npy
        """

        # First, check the inputs
        if (not checkType(graph_name, str)) or (not checkType(graph_dir, str)):
            raise mlPlexusTypeError('Must supply name and directory as str.')

        # Get the class string
        cstr = class_as_string(self)

        # Construct the full path
        graph_path = '{}/{}.{}.npy'.format(graph_dir, cstr, graph_name)

        if checkPath(graph_path):
            raise mlPlexusIOError(
                'The path \'{}\' already exists!'.format(graph_path))
        np.save(graph_path, self.node_attr)

        return graph_path

    def load(self, graph_path):
        """
        Load the Graph Architecture.

        The Graph Architecture can be fully reconstructed from the attribute,
        node_attr.

        Parameters
        ----------
            graph_path: str
                <graph_dir>/mlplexus.graph.GraphArch.<graph_name>.npy
        """

        # First, check the inputs
        if (not checkType(graph_path, str)):
            raise mlPlexusTypeError('Must supply path as str.')
        if not checkPath(graph_path):
            raise mlPlexusIOError(
                'The path \'{}\' does not exists.'.format(graph_path))

        # Get the class string
        cstr = class_as_string(self)

        if cstr not in graph_path:
            raise mlPlexusIOError('The file \'{}\' is invalid for'
                                  ' class \'{}\'.'.format(graph_path, cstr))

        # Re-define the class instance
        self._define_graph(np.load(graph_path, mmap_mode='r'))
