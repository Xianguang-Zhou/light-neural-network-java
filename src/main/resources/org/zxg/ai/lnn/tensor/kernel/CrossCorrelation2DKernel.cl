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
		const int strideH,
		const int strideW,
		const int paddingH,
		const int paddingW,
		const int dilationH,
		const int dilationW,
		const int groups,
		__constant int* inputShape,
		__constant int* weightShape,
		__constant int* resultShape,
		__constant int* inputDimSizes,
		__constant int* weightDimSizes,
		__constant int* resultDimSizes,
		__constant float* input,
		__constant float* weight,
		__global float* result) {
	const size_t gid = get_global_id(0);

	int resultCoordinate[4];
	gidToCoordinate(gid, resultCoordinate, resultDimSizes);

	const int resultGroupSize = resultShape[1] / groups;
	const int groupNumber = resultCoordinate[1] / resultGroupSize;
	const int resultGroupIndex = resultCoordinate[1] % resultGroupSize;
	const int inputGroupSize = weightShape[1];

	int inputCoordinate[4];
	inputCoordinate[0] = resultCoordinate[0];
	const int inputCoordinate1Base = groupNumber * inputGroupSize;
	const int inputCoordinate2Base = resultCoordinate[2] * strideH - paddingH;
	const int inputCoordinate3Base = resultCoordinate[3] * strideW - paddingW;

	int weightCoordinate[4];
	weightCoordinate[0] = resultCoordinate[1];
	weightCoordinate[1] = (resultGroupIndex * inputGroupSize) / resultGroupSize;

	float resultValue = 0;
	const int kernelHeight = weightShape[2];
	const int kernelWidth = weightShape[3];
	for (int inChannelGroupIndex = 0; inChannelGroupIndex < inputGroupSize; ++inChannelGroupIndex) {
		inputCoordinate[1] = inputCoordinate1Base + inChannelGroupIndex;
		for (int kernelHeightIndex = 0; kernelHeightIndex < kernelHeight; ++kernelHeightIndex) {
			inputCoordinate[2] = inputCoordinate2Base + kernelHeightIndex * dilationH;
			weightCoordinate[2] = kernelHeightIndex;
			for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
				inputCoordinate[3] = inputCoordinate3Base + kernelWidthIndex * dilationW;
				float inputValue;
				if (inputCoordinate[2] > -1
					&& inputCoordinate[2] < inputShape[2]
					&& inputCoordinate[3] > -1
					&& inputCoordinate[3] < inputShape[3]) {
					inputValue = input[coordinateToGid(inputCoordinate, inputDimSizes)];
				} else {
					inputValue = 0;
				}

				weightCoordinate[3] = kernelWidthIndex;
				float weightValue = weight[coordinateToGid(weightCoordinate, weightDimSizes)];

				resultValue += (inputValue * weightValue);
			}
		}
	}
	result[gid] = resultValue;
}
