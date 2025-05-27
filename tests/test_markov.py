#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:20:49 2022

@author: dlvilla

Copyright Notice
=================

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC.
Under the terms of Contract DE-NA0003525, there is a non-exclusive license
for use of this work by or on behalf of the U.S. Government.
Export of this program may require a license from the
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS.

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:12:15 2022

@author: dlvilla
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla
"""

from mews.cython import markov_chain
from mews.cython.markov_time_dependent import (
    quadratic_times_exponential_decay_with_cutoff,
)
from mews.stats.markov_time_dependent import (
    quadratic_times_exponential_decay_with_cutoff as quadratic_times_exponential_decay_with_cutoff_py,
)
from mews.stats.markov import MarkovPy

from mews.stats.markov_time_dependent import (
    markov_chain_time_dependent_py,
    markov_chain_time_dependent_wrapper,
)
from mews.stats.extreme import DiscreteMarkov
from numpy.random import default_rng
import os
import numpy as np
import unittest
from matplotlib import pyplot as plt
from matplotlib import rc
from warnings import warn

from time import perf_counter_ns
import logging


class Test_Markov(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()

        cls.file_dir = os.path.join(os.path.dirname(__file__))

        cls.test_weather_path = os.path.join(cls.file_dir, "data_for_testing")
        cls.erase_me_file_path = os.path.join(
            cls.test_weather_path, "erase_me_file.epw"
        )
        if os.path.exists(cls.erase_me_file_path):
            os.remove(cls.erase_me_file_path)

        cls.test_weather_file_path = os.path.join(
            cls.file_dir,
            cls.test_weather_path,
            "USA_NM_Santa.Fe.County.Muni.AP.723656_TMY3.epw",
        )
        cls.tran_mat = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.2, 0.3, 0.4, 0.1],
                [0.3, 0.4, 0.2, 0.1],
            ]
        )
        if cls.plot_results:
            plt.close("all")
            font = {"size": 16}
            rc("font", **font)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.erase_me_file_path):
            os.remove(cls.erase_me_file_path)

    def test_time_dependent(self):

        random_seed = 22305982
        rand_big = default_rng(random_seed).random(100000)
        tran_matrix = np.array(
            [[0.9, 0.02, 0.08], [0.02, 0.98, 0.0], [0.005, 0.0, 0.995]]
        )
        cdf = tran_matrix.cumsum(axis=1)

        # test exponential decay where exponent is so small that this should produce a constant response
        coef_negligible_1term = np.array([[1e-10], [1e-10]])
        coef_1term = np.array([[0.1], [0.1]])

        # these are either a lambda for exponential or slope for linear decay
        # for the first term and a maximum steps term for the second.
        coef_negligible_2terms = np.array([[1e-10, 10000], [1e-10, 10000]])
        coef_2terms = np.array(
            [[0.1, 9], [0.1, 9]]
        )  # should limit steps in a state to 9

        # old constant markov process function - serves as a standard
        state0 = 0
        yy_const = markov_chain(cdf, rand_big[0:1000], state0)

        # loop over python verses cython implementations
        for idx, markov_func in enumerate(
            [markov_chain_time_dependent_py, markov_chain_time_dependent_wrapper]
        ):

            for func_type in range(4):
                ftype = np.array([func_type, func_type], dtype=int)
                if func_type < 2:
                    coef_neg = coef_negligible_1term
                    coef = coef_1term
                else:
                    coef_neg = coef_negligible_2terms
                    coef = coef_2terms

                # a non_constant time decay that is negligible
                yy_negligible = markov_func(cdf, rand_big[0:1000], 0, coef_neg, ftype)

                # a time decay that makes a difference
                yy = markov_func(cdf, rand_big[0:1000], 0, coef, ftype)

                if idx == 0 and func_type == 0:
                    yy_timedecay_py = yy
                elif idx == 1 and func_type == 0:
                    yy_timedecay = yy

                # collect specific results for later
                if func_type == 0:
                    yy_exp = yy
                elif func_type == 1:
                    yy_lin = yy

                # TEST 1 - test negligible decay produces a constant markov process
                self.assertTrue((yy_negligible == yy_const).all())

                # TEST 2 - non-negligible decay does not produce a constant Markov process
                self.assertFalse((yy == yy_const).all())

            # TEST 3 - linear and exponential decays do not produce the same results
            self.assertFalse((yy_exp == yy_lin).all())

        # TEST 4 - python and cython based implementations produce the exact same results
        if not (yy_timedecay_py == yy_timedecay).all():
            warn(
                "The python and cython implementations are working differently."
                + " If you are making changes, you may have forgotten to "
                + "recompile cython! MEWS should do this with python -m setup.py"
                + ". The cython and python versions of markov time dependent chains must"
                + " be synchronized!"
            )
        self.assertTrue((yy_timedecay_py == yy_timedecay).all())

        # TEST 5 cython implementation is faster than python.
        # The input checks take up a LOT of time. making the wrapper 2-3 times faster
        # with input checking off, the implementation is about 10x faster.
        tic_c = perf_counter_ns()
        value = markov_chain_time_dependent_wrapper(
            cdf,
            rand_big[0:8760],
            0,
            coef_1term,
            np.array([np.int32(0), np.int32(0)]),
            check_inputs=False,
        )
        toc_c = perf_counter_ns()

        tic_py = perf_counter_ns()
        value_py = markov_chain_time_dependent_py(
            cdf, rand_big[0:8760], 0, coef_1term, np.array([0, 0])
        )
        toc_py = perf_counter_ns()

        ctime = toc_c - tic_c
        pytime = toc_py - tic_py

        if ctime > pytime:
            warn(
                "WARNING! The cython implementation of 'markov_chain_time_dependent'"
                + "is running slower thant the same python implementation."
            )
        self.assertTrue(ctime < pytime)

        # TEST 6 - very fast decay only allows up to a single time step in different states
        coef_fast_decay = np.array(
            [[100.0, -999], [100.0, 1e7]]
        )  # just checking if different kinds of function types being requested works.
        values = markov_chain_time_dependent_wrapper(
            cdf,
            rand_big,
            0,
            coef_fast_decay,
            np.array([np.int32(0), np.int32(2)]),
            check_inputs=False,
        )
        # If np.diff(values) is state, the next value will be either 0 or -state and the one after it will be -2
        # such a pattern indicates that only 1 hour events are occuring.
        # and then the next values 0
        for state in [1, 2]:
            diffvals = np.diff(values)
            testvals = np.array(
                [
                    ((val1 == 0 and val2 == -state) or (val1 == -state))
                    for idx, (val0, val1, val2) in enumerate(
                        zip(diffvals[:-3], diffvals[1:-2], diffvals[2:])
                    )
                    if val0 == state
                ]
            )
            self.assertTrue(testvals.all())

        # TEST 7 - cutoff causes even a unit matrix to create events of duration
        #          equal to the cutoff
        tran_mat = np.array([[0.9, 0.1], [0.0, 1.0]])
        cdf2 = tran_mat.cumsum(axis=1)

        # BE CAREFUL, changing the 0.0 to 0 will give error ValueError: Buffer dtype mismatch, expected 'DTYPE_t' but got 'long'
        # type checking would correct this but slows downs everything.
        coef_cutoff = np.array([[0.0, 10.0], [0.0, 10.0]])
        for func_type in [2, 3]:
            ftype = np.array([func_type, func_type])
            values = markov_chain_time_dependent_wrapper(
                cdf2, rand_big, 0, coef_cutoff, ftype
            )
            # all changes in state will be of duration 10 + 2
            diffvals = np.diff(values)
            # for python implementation, this is diffvals[:,-12],diffvals[11:] that
            # works. I am not going to try and correct this. It probably has to do
            # with an integer comparison technique that is not properly controlled.
            testvals = np.array(
                [
                    val1 == -1
                    for idx, (val0, val1) in enumerate(
                        zip(diffvals[:-12], diffvals[11:])
                    )
                    if val0 == 1
                ]
            )
            self.assertTrue(testvals.all())

        # # func_type=4
        coef_quadratic = np.array([[100, 0.999, 300], [100, 0.999, 300]])

        values = {}
        for idx, markov_func in enumerate(
            [markov_chain_time_dependent_py, markov_chain_time_dependent_wrapper]
        ):
            values[idx] = markov_func(
                cdf, rand_big[0:1000], 1, coef_quadratic, np.array([4, 4])
            )

        self.assertTrue((values[0] == values[1]).all())

        # func_type=5 delayed exponential with cutoff
        coef_delayed = np.array([[0.001, 300, 100], [0.002, 400, 200]])

        values = {}
        for idx, markov_func in enumerate(
            [markov_chain_time_dependent_py, markov_chain_time_dependent_wrapper]
        ):
            values[idx] = markov_func(
                cdf, rand_big[0:1000], 1, coef_delayed, np.array([5, 5])
            )

        self.assertTrue((values[0] == values[1]).all())

    def test_MarkovChain(self):
        always0 = np.array([[1, 0], [1, 0]], dtype=float)
        state0 = 0
        rand = self.rng.random(10000)

        value = markov_chain(always0, rand, state0)
        value_py = MarkovPy.markov_chain_py(always0, rand, state0)
        np.testing.assert_array_equal(value, value_py)
        self.assertTrue(value_py.sum() == 0.0)

        two_state_equal = np.array([[0.5, 0.5], [0.5, 0.5]])
        rand = self.rng.random(100000)
        value = markov_chain(two_state_equal.cumsum(axis=1), rand, state0)
        value_py = MarkovPy.markov_chain_py(
            two_state_equal.cumsum(axis=1), rand, state0
        )
        np.testing.assert_array_equal(value, value_py)
        # in the limit, the average value should approach 0.5.
        self.assertTrue(np.abs(value.sum() / 100000 - 0.5) < 0.01)

        always1 = np.array([[0, 1], [0, 1]], dtype=float)
        state0 = 1
        rand = self.rng.random(10000)

        value = markov_chain(always1, rand, state0)
        self.assertTrue(value.sum() == 10000)

        # now do a real calculation
        three_state = np.array(
            [[0.9, 0.04, 0.06], [0.5, 0.495, 0.005], [0.5, 0.005, 0.495]]
        )
        val, vec = np.linalg.eig(np.transpose(three_state))
        rand = self.rng.random(int(1e6))

        tic_c = perf_counter_ns()
        value = markov_chain(three_state.cumsum(axis=1), rand, state0)
        toc_c = perf_counter_ns()

        tic_py = perf_counter_ns()
        value_py = MarkovPy.markov_chain_py(three_state.cumsum(axis=1), rand, state0)
        toc_py = perf_counter_ns()

        self.assertAlmostEqual((value - value_py).sum(), 0.0)

        ctime = toc_c - tic_c
        pytime = toc_py - tic_py

        self.assertTrue(ctime < pytime)

        # calculate the steady state via taking the eigenvector of eigenvalue 1
        # eig will handle a left-stochastic matrix so a transpose operation
        # is needed.
        val, vec = np.linalg.eig(np.transpose(three_state))
        steady = vec[:, 0] / vec.sum(axis=0)[0]
        nn = np.histogram(value, bins=[-0.5, 0.5, 1.5, 2.5])
        nn = (nn[0] / nn[0].sum(), nn[1])  # normalize to probability density

        if self.plot_results:
            fig, axl = plt.subplots(1, 2, figsize=(20, 10))
            nn2 = axl[1].hist(value, bins=[-0.5, 0.5, 1.5, 2.5], density=True)
            axl[1].set_ylabel("Fraction of time in state")
            axl[0].plot(value[0:1000])

        # verify loose convergence.
        self.assertTrue(np.linalg.norm(nn[0] - steady) < 0.01)

        logging.log(0, "\n\n")
        logging.log(
            0,
            "For 1e6 random numbers a 3-state Markov"
            + " Chain in python took: {0:5.3e} seconds".format((toc_py - tic_py) / 1e9),
        )
        logging.log(
            0,
            "For 1e6 random numbers a 3-state Markov"
            + " Chain in cython took: {0:5.3e} seconds".format((toc_c - tic_c) / 1e9),
        )

        if pytime / 10 < ctime:
            raise UserWarning(
                "The cython markov_chain function is less than"
                + " 10x faster than native python!"
            )

        state0 = 1  # 2 on wikipedia
        cat_and_mouse = np.array(
            [
                [0, 0, 0.5, 0, 0.5],
                [0, 0, 1, 0, 0],
                [0.25, 0.25, 0, 0.25, 0.25],
                [0, 0, 0.5, 0, 0.5],
                [0, 0, 0, 0, 1.0],
            ],
            dtype=float,
        )
        rand = self.rng.random(1000)

        # this is a markov process which must reach a stationary state of 4
        # no matter what see https://en.wikipedia.org/wiki/Stochastic_matrix
        value = markov_chain(cat_and_mouse.cumsum(axis=1), rand, state0)[-1]

        self.assertEqual(value, 4)

    def test_quadratic_decay_func(self):

        linestyle = [":", "-", "--"]
        color = ["k", "k", "k"]

        nondim_delT = np.arange(0, 10, 0.001)

        Pratio = np.array([0.9, 1.0, 1.1])

        if self.plot_results:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        yy = {}
        yy_py = {}
        for Pr, ls, col in zip(Pratio, linestyle, color):
            yy[Pr] = np.array(
                [
                    quadratic_times_exponential_decay_with_cutoff(
                        nd_delT, 1.0, Pr, 1.0, 9.0
                    )
                    for nd_delT in nondim_delT
                ]
            )
            yy_py[Pr] = np.array(
                [
                    quadratic_times_exponential_decay_with_cutoff_py(
                        nd_delT, 1.0, Pr, 1.0, 9.0
                    )
                    for nd_delT in nondim_delT
                ]
            )

            if self.plot_results:
                ax.plot(
                    nondim_delT,
                    yy[Pr],
                    label=r"$\frac{P_{max_{w,m}}}{P_{0_{w,m}}}$"
                    + "={0:2.1f}".format(Pr),
                    linestyle=ls,
                    color=col,
                )

        if self.plot_results:
            ax.legend()
            ax.set_xlabel(r"$\Delta t / \Delta t_{p_{w,m}}$")
            ax.set_ylabel(r"$P_{s_{w,m}} / P_{0_{w,m}}$")
            ax.grid("on")
            plt.tight_layout()
            plt.savefig("decay_function_illustration.png", dpi=300)

        # remember cutoff time occurs at 9.0
        self.assertAlmostEqual(yy[1.0].sum(), len(yy[1.0]) - 999)

        ind1 = (nondim_delT == 1.0).argmax()
        ind8 = (nondim_delT == 8.0).argmax()
        self.assertAlmostEqual(yy[0.9][ind1], 0.9)
        self.assertAlmostEqual(yy[1.1][ind1], 1.1)
        self.assertAlmostEqual(yy[1.1][ind8], 0.6143431087434705)
        [self.assertAlmostEqual(y_py, y) for y_py, y in zip(yy_py[1.1], yy[1.1])]
        # self.assertTrue((yy_py[1.1]==yy[1.1]).all())


if __name__ == "__main__":
    o = unittest.main(Test_Markov())
