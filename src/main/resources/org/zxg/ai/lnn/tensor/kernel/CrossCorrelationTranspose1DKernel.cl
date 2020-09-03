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
	coordinate[2] = gid % dimSizes[1];
}

int coordinateToGid(int* coordinate, __constant int* dimSizes) {
	return coordinate[0] * dimSizes[0] + coordinate[1] * dimSizes[1] + coordinate[2];
}

__kernel void run(
		const int stride,
		const int padding,
		const int outputPadding,
		const int groups,
		const int dilation,
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

	float resultValue = 0;
	if (resultCoordinate[2] + outputPadding < resultShape[2]) {
		const int resultGroupSize = resultShape[1] / groups;
		const int groupNumber = resultCoordinate[1] / resultGroupSize;
		const int resultGroupIndex = resultCoordinate[1] % resultGroupSize;
		const int inputGroupSize = inputShape[1] / groups;

		int inputCoordinate[3];
		inputCoordinate[0] = resultCoordinate[0];
		const int inputCoordinate1Base = groupNumber * inputGroupSize;
		const int inputCoordinate2Base = resultCoordinate[2] - dilation * (weightShape[2] - 1) + padding;

		int weightCoordinate[3];
		weightCoordinate[1] = resultGroupIndex;

		const int kernelWidth = weightShape[2];
		for (int inChannelGroupIndex = 0; inChannelGroupIndex < inputGroupSize; ++inChannelGroupIndex) {
			inputCoordinate[1] = inputCoordinate1Base + inChannelGroupIndex;
			weightCoordinate[0] = inputCoordinate[1];
			for (int kernelWidthIndex = 0; kernelWidthIndex < kernelWidth; ++kernelWidthIndex) {
				inputCoordinate[2] = inputCoordinate2Base + kernelWidthIndex * dilation;
				float inputValue = 0;
				if (0 == inputCoordinate[2] % stride) {
					inputCoordinate[2] /= stride;
					if (inputCoordinate[2] > -1 && inputCoordinate[2] < inputShape[2]) {
						inputValue = input[coordinateToGid(inputCoordinate, inputDimSizes)];
					}
				}

				weightCoordinate[2] = kernelWidthIndex;
				float weightValue = weight[coordinateToGid(weightCoordinate, weightDimSizes)];

				resultValue += (inputValue * weightValue);
			}
		}
	}
	result[gid] = resultValue;
}
