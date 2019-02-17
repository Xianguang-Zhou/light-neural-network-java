/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor;

import com.aparapi.Kernel;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
class SumAxisKernel extends Kernel {

	@Constant
	int[] sourceDimSizes_$constant$, resultDimSizes_$constant$, sumAxis_$constant$;
	float[] source, result;

	SumAxisKernel(int axis, Tensor source, Tensor result) {
		this.source = source.data;
		this.result = result.data;
		this.sourceDimSizes_$constant$ = source.dimSizes;
		this.resultDimSizes_$constant$ = result.dimSizes;
		this.sumAxis_$constant$ = new int[] { axis, source.shape[axis] };
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid = getGlobalId();
		final int sumAxis = sumAxis_$constant$[0];
		int sourceIndex = 0;
		for (int resultIndex = gid, axis = 0; resultIndex != 0; axis++) {
			if (sumAxis != axis) {
				sourceIndex += ((resultIndex / resultDimSizes_$constant$[axis]) * sourceDimSizes_$constant$[axis]);
			}
			resultIndex %= resultDimSizes_$constant$[axis];
		}
		final int sumAxisDimSize = sourceDimSizes_$constant$[sumAxis];
		for (final int sourceIndexLimit = sourceIndex + sumAxis_$constant$[1]
				* sumAxisDimSize; sourceIndexLimit != sourceIndex; sourceIndex += sumAxisDimSize) {
			result[gid] += source[sourceIndex];
		}
	}
}
