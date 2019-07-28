/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
float nextFloat(long MASK, long MULTIPLIER, long ADDEND, long seed, int index) {
	seed = seed * (index + 1);
	seed = (seed ^ MULTIPLIER) & MASK;
	seed = (seed * MULTIPLIER + ADDEND) & MASK;
	const int next24 = (int) (((unsigned long) seed) >> 24);
	return next24 / ((float) (1 << 24));
}

__kernel void run(
		const long MASK,
		const long MULTIPLIER,
		const long ADDEND,
		const long seed,
		const float mean,
		const float standardDeviation,
		__global float* result) {
	const size_t gid = get_global_id(0);
	if ((gid % 2) == 0) {
		const float u1 = nextFloat(MASK, MULTIPLIER, ADDEND, seed, gid);
		const float u2 = nextFloat(MASK, MULTIPLIER, ADDEND, seed, gid + 1);
		const float z0 = sqrt(-2 * log(u1)) * cos(M_PI_F * 2 * u2);
		result[gid] = z0 * standardDeviation + mean;
	} else {
		const float u1 = nextFloat(MASK, MULTIPLIER, ADDEND, seed, gid - 1);
		const float u2 = nextFloat(MASK, MULTIPLIER, ADDEND, seed, gid);
		const float z1 = sqrt(-2 * log(u1)) * sin(M_PI_F * 2 * u2);
		result[gid] = z1 * standardDeviation + mean;
	}
}
