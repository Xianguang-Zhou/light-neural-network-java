/*
 * Copyright (c) 2021, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
void gidToCoordinate(size_t gid, int* coordinate, __constant int* dimSizes) {
	coordinate[0] = gid / dimSizes[0];
	gid %= dimSizes[0];
	coordinate[1] = gid / dimSizes[1];
	coordinate[2] = gid % dimSizes[1];
}

int coordinateToGid(int* coordinate, __constant int* dimSizes) {
	return coordinate[0] * dimSizes[0] + coordinate[1] * dimSizes[1] + coordinate[2];
}

__kernel void run(
		const int kernelWidth,
		const int stride,
		const int padding,
		const int dilation,
		__constant int* inputShape,
		__constant int* resultShape,
		__constant int* inputDimSizes,
		__constant int* resultDimSizes,
		__constant float* input,
		__global float* result,
		__global int* indices) {
	const size_t gid = get_global_id(0);

	int resultCoordinate[3];
	gidToCoordinate(gid, resultCoordinate, resultDimSizes);

	int inputCoordinate[3];
	inputCoordinate[0] = resultCoordinate[0];
	inputCoordinate[1] = resultCoordinate[1];
	const int inputCoordinate2Base = resultCoordinate[2] * stride - padding;

	float maxValue = 0;
	int maxIndex = -1;
	for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
		inputCoordinate[2] = inputCoordinate2Base + kernelWidthIndex * dilation;
		if (inputCoordinate[2] > -1 && inputCoordinate[2] < inputShape[2]) {
			float value = input[coordinateToGid(inputCoordinate, inputDimSizes)];
			if (maxIndex != -1) {
				if (value > maxValue) {
					maxValue = value;
					maxIndex = inputCoordinate[2];
				}
			} else {
				maxValue = value;
				maxIndex = inputCoordinate[2];
			}
		}
	}
	result[gid] = maxValue;
	if (indices != 0) {
		indices[gid] = maxIndex;
	}
}
