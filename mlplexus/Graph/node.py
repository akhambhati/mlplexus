"""
Implements the Node class in the Graph.

Author: Ankit N. Khambhati
Created: 2018/02/27
Updated: 2018/02/27
"""

from mlplexus.exception import mlPlexusException


class Node(object):
    """
    Base class for a Node in the Graph.

    Node has an immutable identity in the Graph.

    Node can also have variably-named attributes with immutable values.
    """

    def __init__(self, ID=None, **attr):
        """
        Initialize a Node with specific ID and set attributes

        Parameters
        ----------
            ID: str
                Identifier for the node.

            attr: key/value
                Attribute and value assigned to the node
        """

        if ID is None:
            raise mlPlexusException('Node must be assigned an identifier')
        self._ID = ID
        self._attributes = attr

    def __str__(self):
        """Node: <ID>"""
        return 'Node: {}'.format(self._ID)

    def get_ID(self):
        """Give back node ID"""
        return self.ID

    def get_attributes(self):
        """Give back node attributes"""
        return self._attributes
