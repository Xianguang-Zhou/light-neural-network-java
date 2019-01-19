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
class BroadcastKernel extends Kernel {

	@Constant
	int[] sourceDimSizes_$constant$, resultDimSizes_$constant$, sourceShape_$constant$, ndim_$constant$;
	float[] source, result;

	BroadcastKernel(Tensor source, Tensor result) {
		this.sourceDimSizes_$constant$ = source.dimSizes;
		this.resultDimSizes_$constant$ = result.dimSizes;
		this.sourceShape_$constant$ = source.shape;
		this.ndim_$constant$ = new int[] { source.shape.length };
		this.source = source.data;
		this.result = result.data;
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid = getGlobalId();
		final int ndim = ndim_$constant$[0];
		int sourceIndex = 0;
		for (int resultIndex = gid, dimSizesIndex = 0; dimSizesIndex < ndim; dimSizesIndex++) {
			if (sourceShape_$constant$[dimSizesIndex] != 1) {
				sourceIndex += ((resultIndex / resultDimSizes_$constant$[dimSizesIndex])
						* sourceDimSizes_$constant$[dimSizesIndex]);
			}
			resultIndex %= resultDimSizes_$constant$[dimSizesIndex];
		}
		result[gid] = source[sourceIndex];
	}
}
