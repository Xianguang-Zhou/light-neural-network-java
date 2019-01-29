/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.autograd;

import java.util.Arrays;

import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
final class LeftDotComputation extends Computation {

	private final Tensor constant;

	public LeftDotComputation(Variable creator, Tensor constant) {
		super(creator);
		this.constant = constant;
	}

	@Override
	protected Tensor backward(Tensor forwardGradient) {
		if (forwardGradient.ndim() == 0) {
			if (constant.ndim() == 0) {
				return forwardGradient.mul(constant);
			} else {
				return forwardGradient.reshape(1).broadcastTo(constant.shape()).mul(constant);
			}
		} else {
			Tensor creatorValue = creator.value();
			forwardGradient = reduceNdim(forwardGradient, creatorValue.ndim(), true);
			Tensor thisGradient = increaseNdim(reduceNdim(constant, 2, false).transpose(), forwardGradient.ndim());
			Tensor backwardGradient = thisGradient.dot(forwardGradient);
			backwardGradient.setShape(creatorValue.shape());
			return backwardGradient;
		}
	}

	private static Tensor increaseNdim(Tensor tensor, int newNdim) {
		int[] newShape = new int[newNdim];
		int oldNdim = tensor.ndim();
		int[] oldShape = tensor.shape();
		int copyBeginIndex = newNdim - oldNdim;
		for (int i = 0; i < newNdim; i++) {
			if (i < copyBeginIndex) {
				newShape[i] = 1;
			} else {
				newShape[i] = oldShape[i - copyBeginIndex];
			}
		}
		return tensor.reshape(newShape);
	}

	private static Tensor reduceNdim(Tensor tensor, int newNdim, boolean isProductGradient) {
		while (tensor.ndim() > newNdim) {
			int[] tensorShape = tensor.shape();
			int[] constantShape = new int[tensorShape.length];
			int constantShapeLastIndex = constantShape.length - 1;
			constantShape[constantShapeLastIndex] = tensorShape[0];
			for (int i = 0; i < constantShapeLastIndex; i++) {
				constantShape[i] = 1;
			}
			Tensor constantTensor = new Tensor(constantShape);
			float constantValue = 1.0f;
			if (isProductGradient) {
				constantValue /= constantShape[constantShapeLastIndex];
			}
			constantTensor.constant(constantValue);
			tensor = constantTensor.dot(tensor);
			tensor.setShape(Arrays.copyOfRange(tensorShape, 1, tensorShape.length));
		}
		return tensor;
	}
}
