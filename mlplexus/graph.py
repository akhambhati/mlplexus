"""
Implements the basic infrastructure of
Graph, GraphState.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/04/10
"""

import os

import numpy as np

from mlplexus.checks import (checkArrDims, checkArrDTypeStr, checkArrLen,
                             checkArrSqr, checkNone, checkPath, checkType)
from mlplexus.exception import (mlPlexusException, mlPlexusIOError,
                                mlPlexusNotImplemented, mlPlexusTypeError,
                                mlPlexusValueError)
from mlplexus.helper import class_as_string, vprint


class GraphArch(object):
    """
    Class to define the architecture of a multiplexed, complex system.

    Architecture is defined by the following components:
        + Graph:
            - activity: measures of `information' generated at the node,
                        basic working unit of the system.
                        (e.g. brain region, neuron, population of neurons)
            - adjacency: measures of inter-relationship or interaction
                        between pairs of nodes.
        + Layer: contexts in which the system is observed. for example,
            - snapshots of the system in time
            - form of interaction (e.g. structural/functional)
            - mode of interaction (e.g. email/phone/slack/coherence)

    The implementation assumes a persistent identity of nodes across layers.
    No explicit coupling between layers exists.

    Parameters
    ----------
        arch_name: str
            Name of the GraphArch instance (used for cacheing).

        arch_dir: str
            Directory to cache the GraphArch instance.

        node_id: np.ndarray(str,)
            List of strings, where each string identifies a node. Storage of
            immutable node attributes is currently not implemented. IDs should
            follow some legacy naming convention.

        layer_id: list((str, dtype))
            List of tuples, where each string identifies a layer name and each
            dtype denotes the object type that will encode layer description.
    """

    def __init__(self,
                 arch_name=None,
                 arch_dir=None,
                 node_id=None,
                 layer_id=None,
                 verbose=True):
        """Initialize GraphArch with nodes and contextual layers."""

        vprint('\n\n:: Graph Architecture: {} ::'.format(arch_name), verbose)

        # Initial input quality checks
        # Handle the Architecture name and directory
        if (not checkType(arch_name, str)) or (not checkType(arch_dir, str)):
            raise mlPlexusTypeError('Must supply a string for GraphArch name'
                                    ' and directory.')
        cstr_def = '{}.Def'.format(class_as_string(self))
        cstr_dat = '{}.Dat'.format(class_as_string(self))
        arch_def_path = '{}/{}.{}.npy'.format(arch_dir, cstr_def, arch_name)
        arch_dat_path = '{}/{}.{}.npy'.format(arch_dir, cstr_dat, arch_name)

        # Infer the state of the graph definition
        # if not defined, then check all other inputs.
        if not checkPath(arch_def_path):
            vprint('\t- Creating new definition...', verbose)

            # Handle Node IDs
            if not checkType(node_id, np.ndarray):
                raise mlPlexusTypeError('Must supply a node_id'
                                        ' of type np.ndarray.')

            if not checkArrDTypeStr(node_id):
                vprint('\t\t* Warning: Supplied Node IDs'
                       ' should be strings, converting...', verbose)
            else:
                node_id = np.array(node_id, dtype=str)

            if not checkArrDims(node_id, 1):
                vprint('\t\t* Warning: Supplied Node IDs'
                       ' should be flat, converting...', verbose)
            else:
                node_id = node_id.flatten()

            # Handle Layer IDs
            try:
                np.dtype(layer_id)
            except TypeError:
                raise mlPlexusTypeError('Must supply a layer_id'
                                        ' that conforms to structured numpy'
                                        ' array formulation.')

            # Formulate the arch_dtype
            n_node = len(node_id)
            arch_dtype = [('graph', [('activity', np.float64, (n_node)),
                                     ('adjacency', np.float64,
                                      (n_node, n_node))]), ('layers',
                                                            layer_id)]
            try:
                arch_dtype = np.dtype(arch_dtype)
            except TypeError:
                raise mlPlexusTypeError('Unforeseen error creating the dtype'
                                        ' for the GraphArch definition.')

            self._io_arch_def(
                arch_def_path,
                node_id=node_id,
                n_node=n_node,
                arch_dtype=arch_dtype)
        else:
            vprint('\t- Loading existing definition...', verbose)
            self._io_arch_def(arch_def_path)

        # Infer the state of the graph data
        # if not defined, then instantiate
        vprint('\t- Loading dataset...', verbose)
        self._io_arch_dat(arch_dat_path)

    def _io_arch_def(self, path, **save_defs):
        """Input/output the GraphArch Definition"""

        if len([*save_defs]) > 0:
            np.save(path, save_defs)

        save_defs = np.load(path)[()]
        for key, value in save_defs.items():
            self.__setattr__(key, value)

    def _io_arch_dat(self, path):
        """Input/output the GraphArch Data"""

        try:
            self.dataset = np.memmap(path, dtype=self.arch_dtype, mode='r')
            self.data_new = False
        except FileNotFoundError:
            dataset = np.memmap(
                path, dtype=self.arch_dtype, mode='w+', shape=())
            del dataset
            self.dataset = np.memmap(
                path, dtype=self.arch_dtype, mode='r', shape=(0, ))
            self.data_new = True

    def __del__(self):
        """Make sure to check if the Graph Dataset has been written to"""

        if self.data_new:
            try:
                os.remove(self.dataset.filename)
            except Exception as E:
                pass

    def _check_data_input(self, obs_ix, arr_activity, arr_adjacency,
                          **layer_data):
        """ Quality check the dataset inputs """

        # Get a list of all input arrays
        arr_list = [arr_activity, arr_adjacency]
        for key in [*layer_data]:
            arr_list.append(layer_data[key])

        # If obs_ix is none, then appending data --> infer new data size
        curr_len = len(self.dataset)
        if obs_ix is None:
            obs_list = []
            for arr in arr_list:
                if type(arr) is np.ndarray:
                    if arr.ndim > 1:
                        obs_list.append(len(arr))
                    else:
                        obs_list.append(0)
                else:
                    obs_list.append(0)
            n_obs = np.nanmax(obs_list)

            if n_obs > 0:
                new_len = curr_len + n_obs
                obs_ix = np.arange(curr_len, new_len)
            else:
                raise mlPlexusValueError('Cannot infer number of new'
                                         ' observations due to ambiguous'
                                         ' dataset input shapes, try to follow'
                                         ' shape formatting recommendations.')

        # Check obs_ix
        obs_ix = np.array(obs_ix).squeeze()
        if obs_ix.ndim == 0:
            obs_ix = np.expand_dims(obs_ix, axis=0)
        if not checkArrDims(obs_ix, 1):
            raise mlPlexusValueError('obs_ix must be a 1d, vector of indices.')

        # Check arr_activity
        if arr_activity is not None:

            arr_activity = np.array(arr_activity).squeeze()
            if checkArrDims(arr_activity, 1):
                arr_activity = np.expand_dims(arr_activity, axis=0)

            if not ((len(obs_ix), self.n_node) == arr_activity.shape):
                arr_activity = arr_activity.T
                if not ((len(obs_ix), self.n_node) == arr_activity.shape):
                    raise mlPlexusValueError('arr_activity does not have'
                                             ' dimensions (n_obs, n_node).')

        # Check arr_adjacency
        if arr_adjacency is not None:

            arr_adjacency = np.array(arr_adjacency).squeeze()
            if checkArrDims(arr_adjacency, 2):
                if not checkArrSqr(arr_adjacency):
                    raise mlPlexusValueError('arr_adjacency does not have'
                                             ' a square shape.')
                arr_adjacency = np.expand_dims(arr_adjacency, axis=0)

            if not ((len(obs_ix), self.n_node,
                     self.n_node) == arr_adjacency.shape):
                raise mlPlexusValueError(
                    'arr_adjacency does not have'
                    ' dimensions (n_obs, n_node, n_node).')

        # Check layer_data
        for key in [*layer_data]:
            if key not in self.arch_dtype['layers'].names:
                vprint('\t* Warning: Ignoring data for undefined layer {}'
                       .format(key))
                layer_data.pop(key)
                continue

            lr_data = layer_data[key]
            sel_dtype = self.arch_dtype['layers'][key].str
            try:
                lr_data = np.array(lr_data, dtype=sel_dtype).squeeze()
            except TypeError:
                raise mlPlexusTypeError('Data for layer {} cannot be set'
                                        ' as {}.'.format(key, sel_dtype))

            if lr_data.ndim == 0:
                lr_data = np.expand_dims(lr_data, axis=0)
            if not ((len(obs_ix), ) == lr_data.shape):
                raise mlPlexusValueError('Data for layer {} does not have'
                                         ' dimensions (n_obs,).'.format(key))
            layer_data[key] = lr_data

        return obs_ix, arr_activity, arr_adjacency, layer_data

    def modify(self,
               obs_ix=None,
               arr_activity=None,
               arr_adjacency=None,
               **layer_data):
        """
        Modify the entries of the current GraphArch dataset.

        Parameters
        ----------
            obs_ix: np.ndarray(n_obs,)
                Indices corresponding to observations of the Graph.
                If NOT modifying and appending new data,
                then set as None (default).

            arr_activity: np.ndarray(n_obs, n_node)
                Measured node activity for observations, obs_ix.

            arr_adjacency: np.ndarray(n_obs, n_node, n_node)
                Measured node adjacency for observations, obs_ix.

            **layer_data: key --> np.ndarray(n_obs,)
                Labels corresponding to the layer `key`
                for observations, obs_ix.
        """

        # Check the inputs
        obs_ix, arr_activity, arr_adjacency, layer_data = \
            self._check_data_input(obs_ix, arr_activity, arr_adjacency,
                                   **layer_data)

        # Determine the type of modification based on entries in obs_ix
        if len(obs_ix) > 0:

            # Expand dataset if necessary
            curr_len = len(self.dataset)
            max_obs_ix = np.max(obs_ix)
            if max_obs_ix < curr_len:
                self.dataset = np.memmap(
                    self.dataset.filename, dtype=self.arch_dtype, mode='r+')
            else:
                self.dataset = np.memmap(
                    self.dataset.filename,
                    dtype=self.arch_dtype,
                    mode='r+',
                    shape=(max_obs_ix + 1, ))

            # Write the modifications
            if arr_activity is not None:
                self.dataset['graph']['activity'][obs_ix, :] = arr_activity
            if arr_adjacency is not None:
                self.dataset['graph']['adjacency'][
                    obs_ix, :, :] = arr_adjacency
            for key in [*layer_data]:
                self.dataset['layers'][key][obs_ix] = layer_data[key]
            self.dataset.flush()
            self.dataset = np.memmap(
                self.dataset.filename, dtype=self.arch_dtype, mode='r')

        vprint('\t- Modified {} layer observations.'.format(len(obs_ix)))
        self.data_new = False

    def architecture(self):
        """Pretty print a summary of GraphArch Nodes and Layers"""
        raise mlPlexusNotImplemented
