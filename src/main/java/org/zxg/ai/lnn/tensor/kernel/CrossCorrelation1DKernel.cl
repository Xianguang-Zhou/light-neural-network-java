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
	coordinate[2] = gid % dimSizes[1];
}

int coordinateToGid(int* coordinate, __constant int* dimSizes) {
	return coordinate[0] * dimSizes[0] + coordinate[1] * dimSizes[1] + coordinate[2];
}

__kernel void run(
		const int stride,
		const int padding,
		const int dilation,
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

	int resultCoordinate[3];
	gidToCoordinate(gid, resultCoordinate, resultDimSizes);

	const int resultGroupSize = resultShape[1] / groups;
	const int groupNumber = resultCoordinate[1] / resultGroupSize;
	const int resultGroupIndex = resultCoordinate[1] % resultGroupSize;
	const int inputGroupSize = inputShape[1] / groups;

	int inputCoordinate[3], weightCoordinate[3];

	inputCoordinate[0] = resultCoordinate[0];
	inputCoordinate[1] = inputGroupSize * groupNumber + (resultGroupIndex * inputGroupSize) / resultGroupSize;

	weightCoordinate[0] = resultCoordinate[1];

	float resultValue = 0;
	const int kernelWidth = weightShape[2];
	const int inputCoordinate2Base = resultCoordinate[2] * stride - padding;
	for (weightCoordinate[1] = 0; weightCoordinate[1] < weightShape[1]; ++weightCoordinate[1]) {
		for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
			inputCoordinate[2] = inputCoordinate2Base + kernelWidthIndex * dilation;
			float inputValue;
			if (inputCoordinate[2] > -1 && inputCoordinate[2] < inputShape[2]) {
				inputValue = input[coordinateToGid(inputCoordinate, inputDimSizes)];
			} else {
				inputValue = 0;
			}

			weightCoordinate[2] = kernelWidthIndex;
			float weightValue = weight[coordinateToGid(weightCoordinate, weightDimSizes)];

			resultValue += (inputValue * weightValue);
		}
	}
	result[gid] = resultValue;
}
