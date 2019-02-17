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
		if (0 == forwardGradient.ndim()) {
			return constant.mul(forwardGradient.scalar());
		} else if (0 == constant.ndim()) {
			return forwardGradient.mul(constant.scalar());
		} else if (0 == creator.value().ndim()) {
			return forwardGradient.mul(constant).sum();
		} else {
			forwardGradient = changeNdim(forwardGradient, creator.value().ndim(), true);
			Tensor thisGradient = changeNdim(constant, 2, false).transpose();
			return forwardGradient.dot(thisGradient);
		}
	}

	private static Tensor changeNdim(Tensor tensor, int newNdim, boolean isProductGradient) {
		int oldNdim = tensor.ndim();
		if (oldNdim < newNdim) {
			tensor = tensor.expandDims(oldNdim, newNdim - oldNdim);
		} else {
			int axisLengthProduct = 1;
			while (tensor.ndim() > newNdim) {
				int[] tensorShape = tensor.shape();
				int axis = tensorShape.length - 1;
				if (isProductGradient) {
					axisLengthProduct *= tensorShape[axis];
				}
				tensor = tensor.sumAxis(axis);
				tensor.setShape(Arrays.copyOfRange(tensorShape, 0, tensorShape.length - 1));
			}
			if (axisLengthProduct != 1) {
				tensor = tensor.mul(1.0f / axisLengthProduct);
			}
		}
		return tensor;
	}
}
