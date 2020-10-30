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
public class RMSprop extends Optimizer {

	public static final String learningRateOption = "learningRate";
	public static final String epsOption = "eps";
	public static final String momentumOption = "momentum";

	public RMSprop(Iterable<ParamGroup> paramGroups) {
		this(paramGroups, 0.01f);
	}

	public RMSprop(Iterable<ParamGroup> paramGroups, float learningRate) {
		this(paramGroups, learningRate, 1e-8f);
	}

	public RMSprop(Iterable<ParamGroup> paramGroups, float learningRate, float eps) {
		this(paramGroups, learningRate, eps, 0);
	}

	public RMSprop(Iterable<ParamGroup> paramGroups, float learningRate, float eps, float momentum) {
		if (learningRate < 0) {
			throw new LnnException();
		}
		if (eps < 0) {
			throw new LnnException();
		}
		if (momentum < 0) {
			throw new LnnException();
		}
		Map<String, Object> defaults = new HashMap<>();
		defaults.put(learningRateOption, learningRate);
		defaults.put(epsOption, eps);
		defaults.put(momentumOption, momentum);
		init(paramGroups, defaults);
	}

	@Override
	public Variable step(Supplier<Variable> lossSupplier) {
		Variable loss = null == lossSupplier ? null : lossSupplier.get();
		for (ParamGroup paramGroup : paramGroups) {
			float learningRate = (Float) paramGroup.options.get(learningRateOption);
			float eps = (Float) paramGroup.options.get(epsOption);
			float momentumFactor = (Float) paramGroup.options.get(momentumOption);
			float oneSubMomentumFactor = 1 - momentumFactor;
			for (Variable param : paramGroup.params()) {
				if (param.requiresGradient()) {
					Tensor gradient = param.gradient();
					if (0 != momentumFactor) {
						Tensor gradientSum = (Tensor) state.get(param);
						gradientSum = null != gradientSum
								? gradient.mul(gradient).mul(oneSubMomentumFactor).add(gradientSum.mul(momentumFactor))
								: gradient.mul(gradient).mul(oneSubMomentumFactor);
						state.put(param, gradientSum);
						param.value(param.value().sub(gradient.mul(learningRate).div(gradientSum.add(eps).sqrt())));
					} else {
						param.value(param.value()
								.sub(gradient.mul(learningRate).div(gradient.mul(gradient).add(eps).sqrt())));
					}
				}
			}
		}
		return loss;
	}
}
