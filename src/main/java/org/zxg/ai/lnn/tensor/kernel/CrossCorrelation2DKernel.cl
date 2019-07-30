/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
