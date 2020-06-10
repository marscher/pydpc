# This file is part of pydpc.
#
# Copyright 2016 Christoph Wehmeyer
#
# pydpc is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "_core.h":
    void _get_distances(double *points, size_t npoints, size_t ndim, double *distances)
    double _get_kernel_size(double *distances, size_t npoints, double fraction)
    void _get_density(double kernel_size, double *distances, size_t npoints, double *density)
    void _get_delta_and_neighbour(
        double max_distance, double *distances, size_t *order,
        size_t npoints, double *delta, size_t *neighbour)
    void _get_membership(
        size_t *clusters, size_t nclusters, size_t *order, size_t *neighbour, size_t npoints, long *membership)
    void _get_border(
        double kernel_size, double *distances, double *density, long *membership, size_t npoints,
        bool *border_member, double *border_density)
    void _get_halo(
        bool border_only, double *border_density,
        double *density, long *membership, bool *border_member, size_t npoints, int *halo)

def get_distances(np.ndarray[double, ndim=2, mode="c"] points not None):
    npoints = points.shape[0]
    ndim = points.shape[1]
    distances = np.zeros(shape=(npoints, npoints), dtype=np.float64)
    _get_distances(
        <double*> np.PyArray_DATA(points),
        npoints, ndim,
        <double*> np.PyArray_DATA(distances))
    return distances

def get_kernel_size(np.ndarray[double, ndim=2, mode="c"] distances not None, fraction):
    return _get_kernel_size(<double*> np.PyArray_DATA(distances), distances.shape[0], fraction)

def get_density(np.ndarray[double, ndim=2, mode="c"] distances not None, kernel_size):
    npoints = distances.shape[0]
    density = np.zeros(shape=(npoints,), dtype=np.float64)
    _get_density(
        kernel_size,
        <double*> np.PyArray_DATA(distances),
        npoints,
        <double*> np.PyArray_DATA(density))
    return density

def get_delta_and_neighbour(
    np.ndarray[size_t, ndim=1, mode="c"] order not None,
    np.ndarray[double, ndim=2, mode="c"] distances not None,
    max_distance):
    npoints = distances.shape[0]
    delta = np.zeros(shape=(npoints,), dtype=np.float64)
    neighbour = np.zeros(shape=(npoints,), dtype=np.uint)
    _get_delta_and_neighbour(
        max_distance,
        <double*> np.PyArray_DATA(distances),
        <size_t*> np.PyArray_DATA(order),
        npoints,
        <double*> np.PyArray_DATA(delta),
        <size_t*> np.PyArray_DATA(neighbour))
    return delta, neighbour

def get_membership(
    np.ndarray[size_t, ndim=1, mode="c"] clusters not None,
    np.ndarray[size_t, ndim=1, mode="c"] order not None,
    np.ndarray[size_t, ndim=1, mode="c"] neighbour not None):
    npoints = order.shape[0]
    membership = np.zeros(shape=(npoints,), dtype=np.int_)
    _get_membership(
        <size_t*> np.PyArray_DATA(clusters),
        clusters.shape[0],
        <size_t*> np.PyArray_DATA(order),
        <size_t*> np.PyArray_DATA(neighbour),
        npoints,
        <long*> np.PyArray_DATA(membership))
    return membership

def get_border(
    kernel_size,
    np.ndarray[double, ndim=2, mode="c"] distances not None,
    np.ndarray[double, ndim=1, mode="c"] density not None,
    np.ndarray[long, ndim=1, mode="c"] membership not None,
    nclusters):
    npoints = distances.shape[0]
    border_density = np.zeros(nclusters, dtype=np.float64)
    border_member = np.zeros(npoints, dtype=np.bool)
    _get_border(
        kernel_size,
        <double*> np.PyArray_DATA(distances),
        <double*> np.PyArray_DATA(density),
        <long*> np.PyArray_DATA(membership),
        npoints,
        <bool*> np.PyArray_DATA(border_member),
        <double*> np.PyArray_DATA(border_density))
    return border_density, border_member

def get_halo(
    np.ndarray[double, ndim=1, mode="c"] density not None,
    np.ndarray[long, ndim=1, mode="c"] membership not None,
    np.ndarray[double, ndim=1, mode="c"] border_density not None,
    np.ndarray[bool, ndim=1, mode="c"] border_member not None,
    border_only=False):
    halo = membership.astype(np.int32, copy=True)
    _get_halo(
        border_only,
        <double*> np.PyArray_DATA(border_density),
        <double*> np.PyArray_DATA(density),
        <long*> np.PyArray_DATA(membership),
        <bool*> np.PyArray_DATA(border_member),
        density.shape[0],
        <int*> np.PyArray_DATA(halo))
    halo_idx = np.where(halo == -1)[0].astype(np.uint)
    core_idx = np.where(halo != -1)[0].astype(np.uint)
    return halo_idx, core_idx
