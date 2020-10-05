/*
* This file is part of pydpc.
*
* Copyright 2016 Christoph Wehmeyer
*
* pydpc is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "_core.h"

#include <stdlib.h>
#include <math.h>

/***************************************************************************************************
*   static convenience functions (without cython wrappers)
***************************************************************************************************/

static inline double sqr(double x) {
    return (x == 0.0) ? 0.0 : x * x;
}

static double distance(double *points, size_t n, size_t m, size_t ndim) {
    size_t i, o = n * ndim, p = m * ndim;
    double sum = 0.0;
    for (i = 0; i < ndim; ++i)
        sum += sqr(points[o + i] - points[p + i]);
    return sqrt(sum);
}

static int compare_double(const void * a, const void * b) {
    double da = *(const double*) a;
    double db = *(const double*) b;
    return (da > db) - (da < db);
}

/***************************************************************************************************
*   pydpc core functions (with cython wrappers)
***************************************************************************************************/

extern void _get_distances(double *points, size_t npoints, size_t ndim, double *distances) {
    size_t i, j;
    for (i = 0; i < npoints - 1; ++i) {
        size_t o = i * npoints;
        for (j = i + 1; j < npoints; ++j) {
            distances[o + j] = distance(points, i, j, ndim);
            distances[j * npoints + i] = distances[o + j];
        }
    }
}

extern double _get_kernel_size(double *distances, size_t npoints, double fraction) {
    size_t i, j, m = 0, n = (npoints * (npoints - 1)) / 2;
    double kernel_size;
    double *scratch = (double *) malloc(n * sizeof(double));
    for (i = 0; i < npoints - 1; ++i) {
        size_t o = i * npoints;
        for (j = i + 1; j < npoints; ++j)
            scratch[m++] = distances[o + j];
    }
    qsort(scratch, n, sizeof(double), compare_double);
    kernel_size = scratch[(size_t) floor(0.5 + fraction * n)];

    /* kernel_size is used as a divisor, it can't be zero. Fall back to the first
     * nonzero value. Technically, they *could* all be zero (degenerate case
     * where all points are the same), in which case zero will still be returned.
     * Client code checks for this.
     */
    if (kernel_size == 0.0) {
        for (size_t i = 0; i < n; i++) {
            if (scratch[i] != 0.0) {
                kernel_size = scratch[i];
                break;
            }
        }
    }

    free(scratch);
    return kernel_size;
}

extern void _get_density(double kernel_size, double *distances, size_t npoints, double *density) {
    size_t i, j;
    double rho;
    for (i = 0; i < npoints - 1; ++i) {
        size_t o = i * npoints;
        for (j = i + 1; j < npoints; ++j) {
            rho = exp(-sqr(distances[o + j] / kernel_size));
            density[i] += rho;
            density[j] += rho;
        }
    }
}

extern void _get_delta_and_neighbour(
        double max_distance, double *distances, size_t *order, size_t npoints, double *delta, long *neighbour) {
    size_t i, j, o;
    double max_delta = 0.0;
    for (i = 0; i < npoints; ++i) {
        delta[order[i]] = max_distance;
        neighbour[i] = -1;
    }
    delta[order[0]] = -1.0;
    for (i = 1; i < npoints; ++i) {
        o = order[i] * npoints;
        for (j = 0; j < i; ++j) {
            if (distances[o + order[j]] < delta[order[i]]) {
                delta[order[i]] = distances[o + order[j]];
                neighbour[order[i]] = order[j];
            }
        }
        max_delta = (max_delta < delta[order[i]]) ? delta[order[i]] : max_delta;
    }
    delta[order[0]] = max_delta;
}

extern void _get_membership(
        size_t *clusters, size_t nclusters, size_t *order, long *neighbour, size_t npoints, long *membership) {
    size_t i;
    for (i = 0; i < npoints; ++i)
        membership[i] = -1;
    for (i = 0; i < nclusters; ++i)
        membership[clusters[i]] = i;
    for (i = 0; i < npoints; ++i) {
        if (membership[order[i]] == - 1)
            membership[order[i]] = membership[neighbour[order[i]]];
    }
}

extern void _get_border(
        double kernel_size, double *distances, double *density, long *membership, size_t npoints,
        bool *border_member, double *border_density) {
    size_t i, j, o;
    double average_density;
    for (i = 0; i < npoints - 1; ++i) {
        o = i * npoints;
        for (j = i + 1; j < npoints; ++j) {
            if ((membership[i] != membership[j]) && (distances[o + j] < kernel_size)) {
                average_density = 0.5 * (density[i] + density[j]);
                if (border_density[membership[i]] < average_density)
                    border_density[membership[i]] = average_density;
                if (border_density[membership[j]] < average_density)
                    border_density[membership[j]] = average_density;
                border_member[i] = true;
                border_member[j] = true;
            }
        }
    }
}

extern void _get_halo(
        bool border_only, double *border_density,
        double *density, long *membership, bool *border_member, size_t npoints, int *halo) {
    size_t i;
    for (i = 0; i < npoints; ++i) {
        if(density[i] < border_density[membership[i]] && (! border_only || border_member[i])) {
            halo[i] = -1;
        }
    }
}
