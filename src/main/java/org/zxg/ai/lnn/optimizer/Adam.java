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
import org.zxg.ai.lnn.tuple.FloatTuple2;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Adam extends Optimizer {

	public static final String learningRateOption = "learningRate";
	public static final String betasOption = "betas";
	public static final String epsOption = "eps";

	public Adam(Iterable<ParamGroup> paramGroups) {
		this(paramGroups, 1e-3f);
	}

	public Adam(Iterable<ParamGroup> paramGroups, float learningRate) {
		this(paramGroups, learningRate, new FloatTuple2(0.9f, 0.999f));
	}

	public Adam(Iterable<ParamGroup> paramGroups, float learningRate, FloatTuple2 betas) {
		this(paramGroups, learningRate, betas, 1e-8f);
	}

	public Adam(Iterable<ParamGroup> paramGroups, float learningRate, FloatTuple2 betas, float eps) {
		if (learningRate < 0) {
			throw new LnnException();
		}
		if (betas.e0 < 0 || betas.e0 >= 1) {
			throw new LnnException();
		}
		if (betas.e1 < 0 || betas.e1 >= 1) {
			throw new LnnException();
		}
		if (eps < 0) {
			throw new LnnException();
		}
		Map<String, Object> defaults = new HashMap<>();
		defaults.put(learningRateOption, learningRate);
		defaults.put(betasOption, betas);
		defaults.put(epsOption, eps);
		init(paramGroups, defaults);
	}

	@Override
	public Variable step(Supplier<Variable> lossSupplier) {
		Variable loss = null == lossSupplier ? null : lossSupplier.get();
		for (ParamGroup paramGroup : paramGroups) {
			float learningRate = (Float) paramGroup.options.get(learningRateOption);
			FloatTuple2 betas = (FloatTuple2) paramGroup.options.get(betasOption);
			float beta1 = betas.e0;
			float oneSubBeta1 = 1 - beta1;
			float beta2 = betas.e1;
			float oneSubBeta2 = 1 - beta2;
			float eps = (Float) paramGroup.options.get(epsOption);
			for (Variable param : paramGroup.params()) {
				if (param.requiresGradient()) {
					Tensor gradient = param.gradient();
					ParamState paramState = (ParamState) state.get(param);
					if (null == paramState) {
						paramState = new ParamState();
						state.put(param, paramState);
					}
					++paramState.step;
					paramState.gradientSum = null != paramState.gradientSum
							? gradient.mul(oneSubBeta1).add(paramState.gradientSum.mul(beta1))
							: gradient.mul(oneSubBeta1);
					paramState.gradientSquareSum = null != paramState.gradientSquareSum
							? gradient.mul(gradient).mul(oneSubBeta2).add(paramState.gradientSquareSum.mul(beta2))
							: gradient.mul(gradient).mul(oneSubBeta2);
					param.value(param.value()
							.sub(paramState.gradientSum.div((float) (1 - Math.pow(beta1, paramState.step)))
									.mul(learningRate).div(paramState.gradientSquareSum
											.div((float) (1 - Math.pow(beta2, paramState.step))).add(eps).sqrt())));
				}
			}
		}
		return loss;
	}

	public static class ParamState {
		public int step;
		public Tensor gradientSum;
		public Tensor gradientSquareSum;
	}
}
