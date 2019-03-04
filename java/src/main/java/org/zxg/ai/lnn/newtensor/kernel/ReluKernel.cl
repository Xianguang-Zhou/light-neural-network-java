/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(__constant float* source, __global float* result) {
	size_t gid = get_global_id(0);
	float element = source[gid];
	if (element > 0) {
		result[gid] = element;
	} else {
		result[gid] = 0;
	}
}
