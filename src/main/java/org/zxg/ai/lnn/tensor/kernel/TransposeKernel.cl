/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		__constant int* permutation,
		__constant int* sourceDimSizes,
		__constant int* resultDimSizes,
		__constant float* source,
		__global float* result) {
	const size_t gid = get_global_id(0);
	int sourceIndex = 0;
	for (int resultIndex = gid, resultDimSizesIndex = 0; resultIndex != 0;) {
		sourceIndex += ((resultIndex / resultDimSizes[resultDimSizesIndex])
				* sourceDimSizes[permutation[resultDimSizesIndex]]);
		resultIndex %= resultDimSizes[resultDimSizesIndex++];
	}
	result[gid] = source[sourceIndex];
}
