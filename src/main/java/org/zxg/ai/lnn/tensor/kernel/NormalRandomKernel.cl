/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
