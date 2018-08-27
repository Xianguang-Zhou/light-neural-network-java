# -*- coding: utf-8 -*-

# Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tvm
import topi
from tvm.contrib import cblas

from generator import Generator, FunctionSigniture, parseArgs


class NDArray(Generator):

    def __init__(self, dtype, ndim, operator):
        self.dtype = dtype
        self.ndim = ndim
        self.operator = operator

    def shape(self):
        shape = []
        for dim in range(1, self.ndim + 1):
            shape.append(tvm.var('dim' + str(dim)))
        return tuple(shape)

    def compute(self, _args):
        shape = self.shape()
        
        left = tvm.placeholder(shape, dtype=self.dtype, name='left')
        right = tvm.placeholder(shape, dtype=self.dtype, name='right')
        result = self.operator(left, right)

        return FunctionSigniture(result, [left, right, result])

    def functionName(self):
        return self.__class__.__name__ + '_' + self.operator.__name__ + '_' + self.dtype + '_' + str(self.ndim) 


class NDArray_dot(NDArray):
    
    def __init__(self, dtype, ndim):
        NDArray.__init__(self, dtype, ndim, None)
        
    def shape(self):
        leftShape = []
        rightShape = []
        for dim in range(0, self.ndim):
            leftShape.append(tvm.var('dim' + str(dim + 1)))
            rightShape.append(tvm.var('dim' + str(dim + self.ndim)))
        return (leftShape, rightShape)

    def compute(self, args):
        (leftShape, rightShape) = self.shape()
    
        left = tvm.placeholder(tuple(leftShape), dtype=self.dtype, name='left')
        right = tvm.placeholder(tuple(rightShape), dtype=self.dtype, name='right')
        if args.target == 'cuda':
            packedFunction = 'tvm.contrib.cublas.matmul'
        else:
            packedFunction = 'tvm.contrib.cblas.matmul'
        result = tvm.extern(tuple(leftShape[:-1] + rightShape[1:]), [left, right],
               lambda ins, outs: tvm.call_packed(
                   packedFunction,
                   ins[0], ins[1], outs[0], False, False), name='result')

        return FunctionSigniture(result, [left, right, result])

    def isCpu(self, args):
        if args.target == 'cuda':
            return False
        else:
            return True

    def functionName(self):
        return self.__class__.__name__ + '_' + self.dtype + '_' + str(self.ndim) 


def main():
    args = parseArgs()
    dtypeList = ['float32']
    operatorList = [topi.add, topi.subtract, topi.multiply, topi.divide,
                    topi.power, topi.mod]
    for dtype in dtypeList:
        for ndim in range(1, 7):
            if ndim > 1:
                NDArray_dot(dtype, ndim).generate(args)
            for operator in operatorList:
                NDArray(dtype, ndim, operator).generate(args)


if __name__ == '__main__':
    main()

