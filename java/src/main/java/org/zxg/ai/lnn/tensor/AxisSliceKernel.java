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
class AxisSliceKernel extends Kernel {

	@Constant
	int[] slice_$constant$, sourceDimSizes_$constant$, resultDimSizes_$constant$;
	float[] source, result;

	AxisSliceKernel(int axis, int begin, Tensor source, Tensor result) {
		this.slice_$constant$ = new int[] { axis, begin };
		this.source = source.data;
		this.result = result.data;
		this.sourceDimSizes_$constant$ = source.dimSizes;
		this.resultDimSizes_$constant$ = result.dimSizes;
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid = getGlobalId();
		int sourceIndex = 0;
		for (int resultIndex = gid, dimSizesIndex = 0; resultIndex != 0;) {
			sourceIndex += ((resultIndex / resultDimSizes_$constant$[dimSizesIndex])
					* sourceDimSizes_$constant$[dimSizesIndex]);
			resultIndex %= resultDimSizes_$constant$[dimSizesIndex++];
		}
		sourceIndex += (slice_$constant$[1] * sourceDimSizes_$constant$[slice_$constant$[0]]);
		result[gid] = source[sourceIndex];
	}
}
