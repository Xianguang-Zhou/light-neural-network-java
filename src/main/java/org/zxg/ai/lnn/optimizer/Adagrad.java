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
package org.zxg.ai.lnn.optimizer;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import org.zxg.ai.lnn.LnnException;
import org.zxg.ai.lnn.autograd.Variable;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Adagrad extends Optimizer {

	public static final String learningRateOption = "learningRate";
	public static final String epsOption = "eps";

	public Adagrad(Iterable<ParamGroup> paramGroups) {
		this(paramGroups, 0.01f);
	}

	public Adagrad(Iterable<ParamGroup> paramGroups, float learningRate) {
		this(paramGroups, learningRate, 1e-10f);
	}

	public Adagrad(Iterable<ParamGroup> paramGroups, float learningRate, float eps) {
		if (learningRate < 0) {
			throw new LnnException();
		}
		if (eps < 0) {
			throw new LnnException();
		}
		Map<String, Object> defaults = new HashMap<>();
		defaults.put(learningRateOption, learningRate);
		defaults.put(epsOption, eps);
		init(paramGroups, defaults);
	}

	@Override
	public Variable step(Supplier<Variable> lossSupplier) {
		Variable loss = null == lossSupplier ? null : lossSupplier.get();
		for (ParamGroup paramGroup : paramGroups) {
			float learningRate = (Float) paramGroup.options.get(learningRateOption);
			float eps = (Float) paramGroup.options.get(epsOption);
			for (Variable param : paramGroup.params()) {
				if (param.requiresGradient()) {
					Tensor gradient = param.gradient();
					Tensor gradientSum = (Tensor) state.get(param);
					gradientSum = null != gradientSum ? gradient.mul(gradient).add(gradientSum)
							: gradient.mul(gradient);
					state.put(param, gradientSum);
					param.value(param.value().sub(gradient.mul(learningRate).div(gradientSum.add(eps).sqrt())));
				}
			}
		}
		return loss;
	}
}
