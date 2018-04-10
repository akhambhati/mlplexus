"""
Implements the basic infrastructure of
Graph, GraphState.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/04/10
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

    def __init__(self,
                 graph_name=None,
                 graph_dir=None,
                 node_id=None,
                 **node_attr):
        """
        Initialize graph with nodes of specific ID and attributes.

        Parameters
        ----------
            graph_name: str
                Name given to the Graph Architecture (used for cacheing)

            graph_dir: str
                Directory in which the Graph Architecture will be stored.

            node_id: np.ndarray(str,)
                List of strings, where each string identifies a node.

            node_attr: key/value -->
                         value=np.ndarray(obj,)
                Each attribute is defined as a key-value pair.
        """

        # Initial input quality checks
        # Handle the Graph name and directory
        if (not checkType(graph_name, str)) or (not checkType(graph_dir, str)):
            raise mlPlexusTypeError('Must supply a string for graph name'
                                    ' and graph directory.')
        cstr = class_as_string(self)
        graph_path = '{}/{}.{}.npy'.format(graph_dir, cstr, graph_name)

        # If a loadable path exists, load it and ignore all other inputs
        if self._memmap_load(graph_path):
            print('Defining existing Graph Architecture:'
                  ' \'{}\' from \'{}\''.format(graph_name, graph_dir))
            print('   ...other GraphArch inputs ignored...')
            return None

        # If not, then check all the GraphArch constructors
        # Handle Node IDs
        if not checkType(node_id, np.ndarray):
            raise mlPlexusTypeError('Must supply a node_id of type np.ndarray')
        if not checkArrDTypeStr(node_id):
            print('Warning: Supplied Node IDs not strings, converting...')
        node_id = np.array(node_id, dtype=str)
        if not checkArrDims(node_id, 1):
            print('Warning: Supplied Node IDs not flattened, converting...')
        node_id = node_id.flatten()
        n_node = len(node_id)

        # Second, handle the incoming attributes
        if not checkType(node_attr, dict):
            raise mlPlexusTypeError('All attributes must be supplied in'
                                    ' conventional Python `dict` format')
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
        if 'node_id' in [*node_attr]:
            raise mlPlexusValueError('Cannot supply an attribute named:'
                                     ' \'node_id\'.')
        node_attr['node_id'] = node_id

        # All checks passed, now define the graph by opening interface with
        if self._memmap_save(graph_path, node_attr):
            print('Created and Defined Graph Architecture'
                  ' \'{}\' in \'{}\''.format(graph_name, graph_dir))

    def _memmap_load(self, graph_path):
        """Memory map the saved graph"""

        if checkPath(graph_path):
            arr_node_attr = np.load(graph_path, mmap_mode='r')
            self._define_graph(arr_node_attr)
            return True
        else:
            return False

    def _memmap_save(self, graph_path, node_attr):
        """Save the graph"""

        # Edge case, should not enter, if so logic in __init__ is wrong
        if checkPath(graph_path):
            raise mlPlexusIOError(
                'Cannot overwrite existing path \'{}\'.'.format(graph_path))

        dtype_tuple = []
        for key in [*node_attr]:
            key_attr = node_attr[key].squeeze()
            dtype_tuple.append((key, key_attr.dtype, key_attr.shape[1:]))
        n_node = len(node_attr[[*node_attr][0]])

        # Generate a Structured Array from the attributes
        arr_node_attr = np.empty(n_node, dtype=dtype_tuple)
        for key in [*node_attr]:
            arr_node_attr[key] = node_attr[key]

        # Create the file
        np.save(graph_path, arr_node_attr)
        self._memmap_load(graph_path)
        return True

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
