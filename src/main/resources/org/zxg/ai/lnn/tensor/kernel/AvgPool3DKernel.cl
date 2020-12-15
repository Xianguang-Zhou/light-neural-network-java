/*
 * Copyright (c) 2020, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
	gid %= dimSizes[2];
	coordinate[3] = gid / dimSizes[3];
	coordinate[4] = gid % dimSizes[3];
}

int coordinateToGid(int* coordinate, __constant int* dimSizes) {
	return coordinate[0] * dimSizes[0]
		+ coordinate[1] * dimSizes[1]
		+ coordinate[2] * dimSizes[2]
		+ coordinate[3] * dimSizes[3]
		+ coordinate[4];
}

__kernel void run(
		const int kernelDepth,
		const int kernelHeight,
		const int kernelWidth,
		const int strideD,
		const int strideH,
		const int strideW,
		const int paddingD,
		const int paddingH,
		const int paddingW,
		const short countIncludePad,
		const int divisorOverride,
		__constant int* inputShape,
		__constant int* resultShape,
		__constant int* inputDimSizes,
		__constant int* resultDimSizes,
		__constant float* input,
		__global float* result) {
	const size_t gid = get_global_id(0);

	int resultCoordinate[5];
	gidToCoordinate(gid, resultCoordinate, resultDimSizes);

	int inputCoordinate[5];
	inputCoordinate[0] = resultCoordinate[0];
	inputCoordinate[1] = resultCoordinate[1];
	const int inputCoordinate2Base = resultCoordinate[2] * strideD - paddingD;
	const int inputCoordinate3Base = resultCoordinate[3] * strideH - paddingH;
	const int inputCoordinate4Base = resultCoordinate[4] * strideW - paddingW;

	float sum = 0;
	int count = 0;
	for (int kernelDepthIndex = 0; kernelDepthIndex < kernelDepth; ++kernelDepthIndex) {
		inputCoordinate[2] = inputCoordinate2Base + kernelDepthIndex;
		if (inputCoordinate[2] > -1 && inputCoordinate[2] < inputShape[2]) {
			for (int kernelHeightIndex = 0; kernelHeightIndex < kernelHeight; ++kernelHeightIndex) {
				inputCoordinate[3] = inputCoordinate3Base + kernelHeightIndex;
				if (inputCoordinate[3] > -1 && inputCoordinate[3] < inputShape[3]) {
					for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
						inputCoordinate[4] = inputCoordinate4Base + kernelWidthIndex;
						if (inputCoordinate[4] > -1 && inputCoordinate[4] < inputShape[4]) {
							sum += input[coordinateToGid(inputCoordinate, inputDimSizes)];
							if (0 == countIncludePad) {
								++count;
							}
						}
					}
				}
			}
		}
	}
	if (divisorOverride) {
		result[gid] = sum / divisorOverride;
	} else if (countIncludePad) {
		result[gid] = sum / (kernelDepth * kernelHeight * kernelWidth);
	} else {
		if (count) {
			result[gid] = sum / count;
		} else {
			result[gid] = 0;
		}
	}
}
