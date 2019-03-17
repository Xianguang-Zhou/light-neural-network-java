/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		const int ndim,
		__constant int* begin,
		__constant int* sourceDimSizes,
		__constant int* resultDimSizes,
		__constant float* source,
		__global float* result) {
	const size_t gid = get_global_id(0);
	int resultIndex = 0;
	for (int sourceIndex = gid, dimSizesIndex = 0; dimSizesIndex < ndim; dimSizesIndex++) {
		resultIndex += (((sourceIndex / sourceDimSizes[dimSizesIndex]) + begin[dimSizesIndex])
				* resultDimSizes[dimSizesIndex]);
		sourceIndex %= sourceDimSizes[dimSizesIndex];
	}
	result[resultIndex] = source[gid];
}
