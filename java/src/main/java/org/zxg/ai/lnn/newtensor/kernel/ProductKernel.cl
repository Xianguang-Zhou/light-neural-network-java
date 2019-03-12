/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		const int leftAxisLimit,
		const int rightAxisLimit,
		const int cacheHeight,
		__constant int* leftDimSizes,
		__constant int* rightDimSizes,
		__constant int* resultDimSizes,
		__constant float* left,
		__constant float* right,
		__global float* cache,
		__global float* result,
		const int passId) {
	size_t gid0 = get_global_id(0);
	if(0 == passId) {
		size_t gid1 = get_global_id(1);

		int resultDataIndex = gid0;
		int resultDimSizesIndex = 0;
		int resultDimSize = 0;
		int leftDataIndex = gid1;
		for (int i = 0; i < leftAxisLimit;) {
			resultDimSize = resultDimSizes[resultDimSizesIndex++];
			leftDataIndex += ((resultDataIndex / resultDimSize) * leftDimSizes[i++]);
			resultDataIndex %= resultDimSize;
		}
		int rightDataIndex = gid1 * rightDimSizes[0];
		for (int i = 1; i < rightAxisLimit;) {
			resultDimSize = resultDimSizes[resultDimSizesIndex++];
			rightDataIndex += ((resultDataIndex / resultDimSize) * rightDimSizes[i++]);
			resultDataIndex %= resultDimSize;
		}

		cache[gid0 * cacheHeight + gid1] = left[leftDataIndex] * right[rightDataIndex];
	} else {
		float resultValue = 0;
		for (int i = gid0 * cacheHeight, limit = i + cacheHeight; i < limit;) {
			resultValue += cache[i++];
		}
		result[gid0] = resultValue;
	}
}
