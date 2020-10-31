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
public class Adadelta extends Optimizer {

	public static final String rhoOption = "rho";
	public static final String epsOption = "eps";

	public Adadelta(Iterable<ParamGroup> paramGroups) {
		this(paramGroups, 0.9f);
	}

	public Adadelta(Iterable<ParamGroup> paramGroups, float rho) {
		this(paramGroups, rho, 1e-6f);
	}

	public Adadelta(Iterable<ParamGroup> paramGroups, float rho, float eps) {
		if (rho < 0 || rho > 1) {
			throw new LnnException();
		}
		if (eps < 0) {
			throw new LnnException();
		}
		Map<String, Object> defaults = new HashMap<>();
		defaults.put(rhoOption, rho);
		defaults.put(epsOption, eps);
		init(paramGroups, defaults);
	}

	@Override
	public Variable step(Supplier<Variable> lossSupplier) {
		Variable loss = null == lossSupplier ? null : lossSupplier.get();
		for (ParamGroup paramGroup : paramGroups) {
			float rho = (Float) paramGroup.options.get(rhoOption);
			float oneSubRho = 1 - rho;
			float eps = (Float) paramGroup.options.get(epsOption);
			for (Variable param : paramGroup.params()) {
				if (param.requiresGradient()) {
					Tensor gradient = param.gradient();
					Tensor[] paramState = (Tensor[]) state.get(param);
					if (null == paramState) {
						paramState = new Tensor[2];
						state.put(param, paramState);
					}
					Tensor gradientSum = paramState[0];
					gradientSum = null != gradientSum ? gradient.mul(gradient).mul(oneSubRho).add(gradientSum.mul(rho))
							: gradient.mul(gradient).mul(oneSubRho);
					paramState[0] = gradientSum;
					Tensor deltaSum = paramState[1];
					Tensor delta = null != deltaSum ? gradient.mul(deltaSum.add(eps).div(gradientSum.add(eps)).sqrt())
							: gradient.mul(gradientSum.add(eps).reciprocal().mul(eps).sqrt());
					param.value(param.value().sub(delta));
					paramState[1] = null != deltaSum ? delta.mul(delta).mul(oneSubRho).add(deltaSum.mul(rho))
							: delta.mul(delta).mul(oneSubRho);
				}
			}
		}
		return loss;
	}
}
