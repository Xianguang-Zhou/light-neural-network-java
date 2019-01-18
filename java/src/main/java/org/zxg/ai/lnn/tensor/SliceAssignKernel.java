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
class SliceAssignKernel extends Kernel {

	@Constant
	int[] begin_$constant$, sourceDimSizes_$constant$, resultDimSizes_$constant$, ndim_$constant$;
	float[] source, result;

	SliceAssignKernel(int[] begin, Tensor source, Tensor result) {
		this.begin_$constant$ = begin;
		this.source = source.data;
		this.result = result.data;
		this.sourceDimSizes_$constant$ = source.dimSizes;
		this.resultDimSizes_$constant$ = result.dimSizes;
		this.ndim_$constant$ = new int[] { result.dimSizes.length };
	}

	void execute() {
		execute(source.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid = getGlobalId();
		final int ndim = ndim_$constant$[0];
		int resultIndex = 0;
		for (int sourceIndex = gid, dimSizesIndex = 0; dimSizesIndex < ndim; dimSizesIndex++) {
			resultIndex += (((sourceIndex / sourceDimSizes_$constant$[dimSizesIndex]) + begin_$constant$[dimSizesIndex])
					* resultDimSizes_$constant$[dimSizesIndex]);
			sourceIndex %= sourceDimSizes_$constant$[dimSizesIndex];
		}
		result[resultIndex] = source[gid];
	}
}
