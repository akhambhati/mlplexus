"""
Implements the basic infrastructure of Graph, MonoGraph, and MultiGraph.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/02/28
"""

import numpy as np

from mlplexus.checks import (checkArrDims, checkArrDTypeStr, checkNone,
                             checkType)
from mlplexus.exception import (mlPlexusException, mlPlexusNotImplemented,
                                mlPlexusTypeError)


class Graph(object):
    """
    Base class for the Graph; representions and manipulations are pure numpy.

    Graph representation of the system is a function of:
        1) Interconnected topology between the nodes
        2) State-based attributes of the individual nodes

    Constraints within the Graph object:
        1) All nodes have identical types of attributes (values can differ).
        2) All edges represent the same mode of interaction.
        3) Two nodes can only have one edge between them.
        3) Edges are always weighted.
        4) Nodes can not have self-loops.
    """

    def __init__(self):
        raise mlPlexusNotImplemented


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
                         value=np.ndarray(obj,)
                Each attribute is defined as a key-value pair.
                    Use `key=None` when defining key as a placeholder.
                    Use `key=obj` when defining same key-value across nodes.
                    Use `key=np.ndarray(obj,)` when defining different
                        key-value across nodes.
        """

        # First, handle the node_id
        if checkNone(node_id) or (not checkType(node_id, np.ndarray)):
            raise mlPlexusTypeError('Must supply a node_id of type np.ndarray')
        if not checkArrDims(node_id, 1):
            raise mlPlexusException(
                'Must supply a node_id with a 1d-array of identifiers')
        if not checkArrDTypeStr(node_id):
            raise mlPlexusTypeError(
                'Must supply a node_id containing strings as identifiers')
        self._id = node_id
        n_node = len(self._id)

        # Second, handle the incoming attributes
        if not checkType(node_attr, dict):
            raise mlPlexusTypeError('All attributes must be supplied in'
                                    ' conventional Python `dict` format')
        self._attr_key = np.array([*node_attr])
        n_attr = len(self._attr_key)

        # Parse the attributes into a LUT
        self._attr_arr = np.nan * np.zeros((n_attr, n_node))
        if not bool(node_attr):
            return

        for key, value in node_attr.items():
            if checkNone(value):
                continue
            if checkType(value, list):
                raise mlPlexusTypeError('Cannot supply list as value for'
                                        ' attribute {}. Try a 1d-array.'
                                        .format(key))
            if checkType(value, np.ndarray):
                if not checkArrDims(value, 1):
                    raise mlPlexusTypeError('Must supply a 1d-array'
                                            ' of values for attribute {}.'
                                            .format(key))
                if len(value) != self.n_node:
                    raise mlPlexusTypeError('Must either supply one value'
                                            ' or a 1d-array of values equal'
                                            ' in length to number of nodes'
                                            ' for attribute {}.'.format(key))

            # Passed checks for properly formatted value
            k_ix = np.flatnonzero(self._attr_key == key)[0]
            self._attr_arr[k_ix, :] = value

    def __str__(self):
        """Prettified string representation of node ids"""
        s = '\n::NODES::\n'
        return s + '\n'.join(self._id)

    def get_id(self):
        """Return the node IDs"""
        return self._id

    def get_attr_key(self):
        """Return keys defining attributes for the nodes"""
        return self._attr_key

    def get_attr_val(self, key=None):
        """Return node attribute values corresponding to a set of keys"""
        if checkNone(key):
            return self._attr_arr
        if checkType(key, list) or checkType(key, np.ndarray):
            raise mlPlexusTypeError('Cannot supply keys as array-like lists.')
        if key in self._attr_key:
            k_ix = np.flatnonzero(self._attr_key == key)[0]
            return self._attr_arr[k_ix, :]
        else:
            raise mlPlexusTypeError('Key {} is not a valid attribute.'
                                    .format(key))
