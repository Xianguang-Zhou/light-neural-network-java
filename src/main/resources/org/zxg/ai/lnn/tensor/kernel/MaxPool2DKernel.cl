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
	gid %= dimSizes[1];
	coordinate[2] = gid / dimSizes[2];
	coordinate[3] = gid % dimSizes[2];
}

int coordinateToGid(int* coordinate, __constant int* dimSizes) {
	return coordinate[0] * dimSizes[0]
		+ coordinate[1] * dimSizes[1]
		+ coordinate[2] * dimSizes[2]
		+ coordinate[3];
}

__kernel void run(
		__constant float* input,
		__constant int* inputShape,
		__constant int* inputDimSizes,
		__global float* result,
		__constant int* resultDimSizes,
		__global int* indices,
		const int kernelHeight,
		const int kernelWidth,
		const int strideH,
		const int strideW,
		const int paddingH,
		const int paddingW,
		const int dilationH,
		const int dilationW) {
	const size_t gid = get_global_id(0);

	int resultCoordinate[4];
	gidToCoordinate(gid, resultCoordinate, resultDimSizes);

	int inputCoordinate[4];
	inputCoordinate[0] = resultCoordinate[0];
	inputCoordinate[1] = resultCoordinate[1];
	const int inputCoordinate2Base = resultCoordinate[2] * strideH - paddingH;
	const int inputCoordinate3Base = resultCoordinate[3] * strideW - paddingW;

	float maxValue = 0;
	int maxIndex = -1;
	for (int kernelHeightIndex = 0; kernelHeightIndex < kernelHeight; ++kernelHeightIndex) {
		inputCoordinate[2] = inputCoordinate2Base + kernelHeightIndex * dilationH;
		if (inputCoordinate[2] > -1 && inputCoordinate[2] < inputShape[2]) {
			for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
				inputCoordinate[3] = inputCoordinate3Base + kernelWidthIndex * dilationW;
				if (inputCoordinate[3] > -1 && inputCoordinate[3] < inputShape[3]) {
					int inputGid = coordinateToGid(inputCoordinate, inputDimSizes);
					float value = input[inputGid];
					if (maxIndex != -1) {
						if (value > maxValue) {
							maxValue = value;
							maxIndex = inputGid;
						}
					} else {
						maxValue = value;
						maxIndex = inputGid;
					}
				}
			}
		}
	}
	result[gid] = maxValue;
	if (indices != 0) {
		indices[gid] = maxIndex;
	}
}
