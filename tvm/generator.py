# -*- coding: utf-8 -*-

# Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import argparse
import tvm
from tvm.contrib import cc


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='llvm', help='Value: "llvm", "opencl" or "cuda".')
    parser.add_argument('-dme', '--deviceModuleExtension', default='.cl', help='Value: ".cl" or ".ptx".')
    parser.add_argument('-f', '--factor', type=int, default=0, help='Integer value, for example: "64".')
    parser.add_argument('-el', '--exportLibrary', action="store_true", help='Export library.')
    parser.add_argument('-th', '--targetHost', default='llvm', help='Value: "llvm".')
    return parser.parse_args()


class FunctionSigniture:

    def __init__(self, result, arguments):
        self.result = result
        self.arguments = arguments


class Generator:

    def generate(self, args):
        functionName = self.functionName()
        
        if os.path.exists(functionName + '.so'):
            return
        
        signiture = self.compute(args)

        isCpu = self.isCpu(args)

        schedule = tvm.create_schedule(signiture.result.op)
        if args.factor > 0 and not isCpu:
            bx, tx = schedule[signiture.result].split(signiture.result.op.axis[0], args.factor)
            schedule[signiture.result].bind(bx, tvm.thread_axis('blockIdx.x'))
            schedule[signiture.result].bind(tx, tvm.thread_axis('threadIdx.x'))

        if isCpu:
            buildTarget = 'llvm'
        else:
            buildTarget = args.target
        function = tvm.build(schedule, signiture.arguments, buildTarget, target_host=args.targetHost, name=functionName)
        
        if args.exportLibrary:
            function.export_library(functionName + '.so')
        else:
            function.save(functionName + '.o')
            if args.factor > 0 and not isCpu:
                function.imported_modules[0].save(functionName + args.deviceModuleExtension)
            cc.create_shared(functionName + '.so', [functionName + '.o'])

    def isCpu(self, _args):
        return False

    def compute(self):
        pass

    def functionName(self):
        pass

