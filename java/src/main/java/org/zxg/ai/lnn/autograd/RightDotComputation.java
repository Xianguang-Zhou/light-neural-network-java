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
final class RightDotComputation extends Computation {

	private final Tensor constant;

	public RightDotComputation(Variable creator, Tensor constant) {
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
			Tensor backwardGradient = forwardGradient.dot(thisGradient);
			backwardGradient.setShape(creatorValue.shape());
			return backwardGradient;
		}
	}

	private static Tensor increaseNdim(Tensor tensor, int newNdim) {
		int[] newShape = new int[newNdim];
		int oldNdim = tensor.ndim();
		int[] oldShape = tensor.shape();
		for (int i = 0; i < newNdim; i++) {
			if (i < oldNdim) {
				newShape[i] = oldShape[i];
			} else {
				newShape[i] = 1;
			}
		}
		return tensor.reshape(newShape);
	}

	private static Tensor reduceNdim(Tensor tensor, int newNdim, boolean isProductGradient) {
		while (tensor.ndim() > newNdim) {
			int[] tensorShape = tensor.shape();
			int[] constantShape = new int[tensorShape.length];
			constantShape[0] = tensorShape[tensorShape.length - 1];
			for (int i = 1; i < constantShape.length; i++) {
				constantShape[i] = 1;
			}
			Tensor constantTensor = new Tensor(constantShape);
			float constantValue = 1.0f;
			if (isProductGradient) {
				constantValue /= constantShape[0];
			}
			constantTensor.constant(constantValue);
			tensor = tensor.dot(constantTensor);
			tensor.setShape(Arrays.copyOfRange(tensorShape, 0, tensorShape.length - 1));
		}
		return tensor;
	}
}
