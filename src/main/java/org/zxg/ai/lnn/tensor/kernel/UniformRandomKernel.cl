/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
__kernel void run(
		const long MASK,
		const long MULTIPLIER,
		const long ADDEND,
		long seed,
		const float low,
		const float interval,
		__global float* result) {
	const size_t gid = get_global_id(0);
	seed = seed * (gid + 1);
	seed = (seed ^ MULTIPLIER) & MASK;
	seed = (seed * MULTIPLIER + ADDEND) & MASK;
	const int next24 = (int) (((unsigned long) seed) >> 24);
	const float nextFloat = next24 / ((float) (1 << 24));
	result[gid] = nextFloat * interval + low;
}
