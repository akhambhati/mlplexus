"""
Implements the basic infrastructure of
Nodes, BaseGraph, MonoGraph, and MultiGraph.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/02/28
"""

import numpy as np

from mlplexus.checks import (checkArrDims, checkArrDTypeStr, checkArrSqr,
                             checkNone, checkType)
from mlplexus.exception import (mlPlexusException, mlPlexusNotImplemented,
                                mlPlexusTypeError, mlPlexusValueError)


class Nodes(object):
    """
    Base class to establish a collection of Nodes.

    Nodes is a collection of individual nodes, each with an
    immutable identity in the system.

    *NOTE*: Any mutable attributes of Nodes are identified in objects
    derived from the Graph class, _not_ in the Nodes class.
    For example:
        For an instance of Nodes defining a collection of
        humans, dogs, and cats might have a common attribute type
        called `Animal`, which would reasonably remain immutable
        regardless of when/where/how the nodes are observed.

        Alternatively, for an instance of Nodes defining a collection of
        humans with a common attribute type called `Location', might be
        immutable if we expect humans to remain at the same spot in the system.
        If we do not assume they will stay in the same spot in the system,
        then `Location' should not be defined in the Nodes instance.
    """

    def __init__(self, node_id=None, **node_attr):
        """
        Initialize Nodes with specific ID and set attributes

        Parameters
        ----------
            node_id: np.ndarray(str,)
                List of strings, where each string identifies a node.

            node_attr: key/value -->
                         value=None             OR
                         value=obj              OR
                         value=list(obj,)
                Each attribute is defined as a key-value pair.
                    Use `key=None` when defining key as a placeholder.
                    Use `key=obj` when defining same key-value across nodes.
                    Use `key=list(obj,)` when defining different
                        key-value across nodes.
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
        self._id = node_id
        self.n_node = len(self._id)

        # Second, handle the incoming attributes
        if not checkType(node_attr, dict):
            raise mlPlexusTypeError('All attributes must be supplied in'
                                    ' conventional Python `dict` format')

        # Check how number of values compares to number of nodes
        for key in [*node_attr]:
            if checkType(node_attr[key], np.ndarray):
                node_attr[key] = list(node_attr[key])
            if checkType(node_attr[key], list):
                if len(node_attr[key]) != self.n_node:
                    raise mlPlexusTypeError('Must supply same number of values'
                                            ' as nodes for attribute'
                                            ' \'{}\'.'.format(key))
            else:
                node_attr[key] = [node_attr[key] for i in range(self.n_node)]

        # Passed checks for properly formatted value
        self._node_attr = node_attr

    def __str__(self):
        """Prettified string representation of node ids."""
        s = '\n::NODES::\n'
        return s + '\n'.join(self._id)

    def get_id(self):
        """Return the node IDs."""
        return self._id

    def get_attr_key(self):
        """Return keys defining attributes for the nodes."""
        return [*self._node_attr]

    def get_attr_val(self, key=None):
        """Return node attribute values corresponding to a key."""
        if checkNone(key):
            raise mlPlexusException('A node attribute key was not entered.')
        if checkType(key, list) or checkType(key, np.ndarray):
            raise mlPlexusTypeError('Cannot supply keys as array-like lists.')
        if key in [*self._node_attr]:
            return self._node_attr[key]
        else:
            raise mlPlexusTypeError('Key {} is not a valid attribute.'
                                    .format(key))


class Graph(object):
    """
    Base class to establish the Graph object; implementation
    is pure python, numpy.

    Graph provides a flexible format define the
    topology of edges between the nodes.

    Graph defines the topology of edges between the nodes as a function
    for an arbitrary, user-defined complex system state.

    Constraints within the Graph object:
        1) Nodes are predefined in an instance of Nodes class.
        2) Two nodes can only have one edge between them.
        3) Mode of edge interaction is completely described
           by the state-definition. Edge type is not explicitly defined.
    """

    def __init__(self, nodes=None, A=None, **graph_state):
        """
        Initialize Graph with an instance of Nodes, a specification of
        edges interlinking the nodes in Nodes, and a state definition.

        Parameters
        ----------
            nodes: mlplexus.graph.Nodes
                An instance of defined nodes with node attributes.
            A: np.ndarray, shape: (n_node, n_node)
                Adjacency matrix that defines the interlinks between nodes.
                Matrix should a size equal in size to the number of nodes.
            graph_state: key/value -->
                         value=None             OR
                         value=obj
                Each state attribute is defined as a key-value pair.
                    Use `key=None` when defining key as a placeholder.
                    Use `key=obj` when defining key with any hashable object.
        """

        # First, handle nodes
        if checkNone(nodes) or (not checkType(nodes, Nodes)):
            raise mlPlexusTypeError('Must supply an instance of'
                                    ' mlplexus.graph.Nodes')
        self.nodes = nodes

        # Second, handle A
        if not checkType(A, np.ndarray):
            raise mlPlexusTypeError('Must supply an Adjacency matrix'
                                    ' as a square, numpy.ndarray.')
        if not checkArrDims(A, 2):
            raise mlPlexusException('Must supply an Adjacency matrix'
                                    ' with square, 2d-array.')
        if not checkArrSqr(A):
            raise mlPlexusException('Must supply an Adjacency matrix'
                                    ' with square, 2d-array.')
        if not len(A) == self.nodes.n_node:
            raise mlPlexusException('Must supply an Adjacency matrix'
                                    ' with size equal to nodes.')
        if np.isnan(A).any():
            raise mlPlexusValueError('Please resolve NaN in Adjacency matrix.')

        # Infer graph properties from structure of adjacency matrix
        self.graph_props = {
            'directed': not np.array_equal(A, A.T),
            'binarized': np.array_equal(A, A.astype(bool)),
            'loops': np.diag(A > 0).any()
        }
        self._A_to_Ahat(A)

        # Third, handle the incoming state-definition
        if not checkType(graph_state, dict):
            raise mlPlexusTypeError('All state-defs must be supplied in'
                                    ' conventional Python `dict` format')
        self._graph_state = graph_state

    def _A_to_Ahat(self, A):
        """Use graph properties to convert adj. matr. to cfg. vec"""
        if hasattr(self, '_Ahat'):
            return 0

        ix, iy = np.triu_indices_from(A, k=1)
        if self.graph_props['directed']:
            tril = np.tril_indices_from(A, k=-1)
            ix = np.concatenate((ix, tril[0]))
            iy = np.concatenate((iy, tril[1]))
        if self.graph_props['loops']:
            diag = np.diag_indices_from(A)
            ix = np.concatenate((ix, diag[0]))
            iy = np.concatenate((iy, diag[1]))
        self._trans_ix = ix
        self._trans_iy = iy

        self.Ahat = A[self._trans_ix, self._trans_iy]

    def get_adjacency_matrix(self):
        """Return the matrix representation of the Graph."""
        A = np.zeros((self.nodes.n_node, self.nodes.n_node))
        A[self._trans_ix, self._trans_iy] = self.Ahat
        if not self.graph_props['directed']:
            A += np.triu(A, k=1).T
        return A

    def get_configuration_vector(self):
        """Return the vector representation of the Graph."""
        return self.Ahat
