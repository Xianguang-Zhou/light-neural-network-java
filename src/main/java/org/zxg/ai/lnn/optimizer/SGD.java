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
public class SGD extends Optimizer {

	public static final String learningRateOption = "learningRate";
	public static final String momentumOption = "momentum";
	public static final String nesterovOption = "nesterov";

	public SGD(Iterable<ParamGroup> paramGroups, float learningRate) {
		this(paramGroups, learningRate, 0);
	}

	public SGD(Iterable<ParamGroup> paramGroups, float learningRate, float momentum) {
		this(paramGroups, learningRate, momentum, false);
	}

	public SGD(Iterable<ParamGroup> paramGroups, float learningRate, float momentum, boolean nesterov) {
		if (learningRate < 0) {
			throw new LnnException();
		}
		if (momentum < 0) {
			throw new LnnException();
		}
		if (nesterov && 0 == momentum) {
			throw new LnnException();
		}
		Map<String, Object> defaults = new HashMap<>();
		defaults.put(learningRateOption, learningRate);
		defaults.put(momentumOption, momentum);
		defaults.put(nesterovOption, nesterov);
		init(paramGroups, defaults);
	}

	@Override
	public Variable step(Supplier<Variable> lossSupplier) {
		Variable loss = null == lossSupplier ? null : lossSupplier.get();
		for (ParamGroup paramGroup : paramGroups) {
			float learningRate = (Float) paramGroup.options.get(learningRateOption);
			float momentumFactor = (Float) paramGroup.options.get(momentumOption);
			boolean nesterov = (Boolean) paramGroup.options.get(nesterovOption);
			for (Variable param : paramGroup.params()) {
				if (param.requiresGradient()) {
					Tensor deltaParam = param.gradient();
					if (0 != momentumFactor) {
						Tensor lastMomentum = (Tensor) state.get(param);
						lastMomentum = null != lastMomentum ? deltaParam.add(lastMomentum.mul(momentumFactor))
								: deltaParam;
						state.put(param, lastMomentum);
						deltaParam = nesterov ? deltaParam.add(lastMomentum.mul(momentumFactor)) : lastMomentum;
					}
					param.value(param.value().sub(deltaParam.mul(learningRate)));
				}
			}
		}
		return loss;
	}
}
