/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		const int cacheLength,
		__constant float* left,
		__constant float* right,
		__global float* cache,
		__global float* result,
		const int passId) {
	if(0 == passId) {
		size_t gid = get_global_id(0);
		cache[gid] = left[gid] * right[gid];
	} else {
		float resultValue = 0;
		for (int i = 0; i < cacheLength;) {
			resultValue += cache[i++];
		}
		result[0] = resultValue;
	}
}
