/*
 * Copyright (c) 2019, 2020, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
