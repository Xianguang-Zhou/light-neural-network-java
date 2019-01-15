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
class TransposeKernel extends Kernel {

	@Constant
	int[] permutation_$constant$, sourceDimSizes_$constant$, resultDimSizes_$constant$;
	float[] source, result;

	TransposeKernel(int[] permutation, Tensor source, Tensor result) {
		this.permutation_$constant$ = permutation;
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
		for (int resultIndex = gid, resultDimSizesIndex = 0; resultIndex != 0;) {
			sourceIndex += ((resultIndex / resultDimSizes_$constant$[resultDimSizesIndex])
					* sourceDimSizes_$constant$[permutation_$constant$[resultDimSizesIndex]]);
			resultIndex %= resultDimSizes_$constant$[resultDimSizesIndex++];
		}
		result[gid] = source[sourceIndex];
	}
}
