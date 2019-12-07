/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
package org.zxg.ai.lnn.autograd;

import org.zxg.ai.lnn.opencl.IntArray;
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
		if (0 == forwardGradient.ndim()) {
			return constant.mul(forwardGradient.scalar());
		} else if (0 == constant.ndim()) {
			return forwardGradient.mul(constant.scalar());
		} else if (0 == creator.value().ndim()) {
			return forwardGradient.mul(constant).sum();
		} else {
			forwardGradient = changeNdim(forwardGradient, creator.value().ndim(), true);
			Tensor thisGradient = changeNdim(constant, 2, false).transpose();
			return thisGradient.dot(forwardGradient);
		}
	}

	private static Tensor changeNdim(Tensor tensor, int newNdim, boolean isProductGradient) {
		int oldNdim = tensor.ndim();
		if (oldNdim < newNdim) {
			tensor = tensor.expandDims(0, newNdim - oldNdim);
		} else {
			int axisLengthProduct = 1;
			while (tensor.ndim() > newNdim) {
				IntArray tensorShape = tensor.shape();
				if (isProductGradient) {
					axisLengthProduct *= tensorShape.get(0);
				}
				tensor = tensor.sumAxis(0);
				tensor.setShape(IntArray.copyOfRange(tensorShape, 1, tensorShape.length));
			}
			if (axisLengthProduct != 1) {
				tensor = tensor.mul(1.0f / axisLengthProduct);
			}
		}
		return tensor;
	}
}
