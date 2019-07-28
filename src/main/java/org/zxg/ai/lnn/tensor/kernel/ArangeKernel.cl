/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		const float start,
		const float stop,
		const float step,
		const int repeat,
		__global float* result) {
    size_t gid = get_global_id(0);
    float value = start + (gid / repeat) * step;
    if (value < stop) {
    	result[gid] = value;
    }
}
