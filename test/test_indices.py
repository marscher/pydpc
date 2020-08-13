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
import unittest

import numpy as np
from numpy.testing import assert_array_less

from pydpc import Cluster


class TestFourGaussians2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # data generation
        state = np.random.RandomState(1131325452)
        state = np.random.RandomState(0)
        cls.npoints = 1000
        cls.mux = 1.8
        cls.muy = 1.8
        cls.fraction = 0.02
        cls.points = np.empty(shape=(cls.npoints, 2), dtype=np.float64)
        cls.points[:, 0] = state.randn(cls.npoints) + \
                           cls.mux * (-1) ** state.randint(0, high=2, size=cls.npoints)
        cls.points[:, 1] = state.randn(cls.npoints) + \
                       cls.muy * (-1) ** state.randint(0, high=2, size=cls.npoints)
        # cluster initialisation
        cls.dpc = Cluster(cls.points, cls.fraction, autoplot=False)
        cls.dpc.assign(20, 1.5)

    def test_order(self):
        assert_array_less(-1, self.dpc.order)
        assert_array_less(self.dpc.order, self.npoints)

    def test_neighbour(self):
        assert_array_less(-2, self.dpc.neighbour)
        assert_array_less(self.dpc.neighbour, self.npoints)
        assert (self.dpc.neighbour == -1).sum() == 1

    def test_clusters(self):
        assert_array_less(-2, self.dpc.clusters)
        assert_array_less(self.dpc.clusters, self.npoints)

    def test_membership(self):
        assert_array_less(-1, self.dpc.membership)
        assert_array_less(self.dpc.membership, self.dpc.nclusters)

    def test_halo_idx(self):
        assert_array_less(-1, self.dpc.halo_idx)
        assert_array_less(self.dpc.halo_idx, self.npoints)

    def test_core_idx(self):
        assert_array_less(-1, self.dpc.core_idx)
        assert_array_less(self.dpc.core_idx, self.npoints)


if __name__ == '__main__':
    unittest.main()
