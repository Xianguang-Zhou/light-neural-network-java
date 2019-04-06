/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.autograd;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Variable {

	private Tensor value;
	private Tensor gradient;
	private List<Computation> computations;
	private boolean requiresGradient;

	public Variable(Tensor value) {
		this(value, true);
	}

	public Variable(Tensor value, boolean requiresGradient) {
		this.value = value;
		this.requiresGradient = requiresGradient;
		if (requiresGradient) {
			this.gradient = new Tensor(value.shape());
		}
	}

	protected Variable(Tensor value, Computation... computations) {
		this.value = value;
		this.requiresGradient = false;
		if (computations != null) {
			List<Computation> computaionList = new LinkedList<>();
			for (Computation c : computations) {
				if (c.creator.requiresGradient) {
					computaionList.add(c);
				}
			}
			if (!computaionList.isEmpty()) {
				this.requiresGradient = true;
				this.computations = computaionList;
			}
		}
	}

	private static final class BackwardContext {
		final Tensor forwardGradient;
		final Computation computation;

		BackwardContext(Tensor forwardGradient, Computation computation) {
			this.forwardGradient = forwardGradient;
			this.computation = computation;
		}
	}

	public void backward(Tensor gradient) {
		if (null == gradient) {
			gradient = new Tensor(this.value.shape());
			gradient.ones();
		} else {
			value.checkSameShape(gradient);
		}
		Deque<BackwardContext> contextStack = new LinkedList<>();
		contextStack.push(new BackwardContext(gradient, new OnesGradientComputation(this)));
		while (!contextStack.isEmpty()) {
			BackwardContext context = contextStack.pop();
			Computation contextComputation = context.computation;
			Tensor variableGradient = contextComputation.backward(context.forwardGradient);
			Variable variable = contextComputation.creator;
			if (variable.computations != null) {
				for (Computation c : variable.computations) {
					contextStack.push(new BackwardContext(variableGradient, c));
				}
			} else if (variable.gradient != null) {
				variable.gradient = variable.gradient.add(variableGradient);
			}
		}
	}

	public void backward() {
		backward(null);
	}

	public Variable reshape(int... shape) {
		return new Variable(value.reshape(shape), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.reshape(Variable.this.value.shape());
			}
		});
	}

	public Variable expandDims(int axis, int times) {
		return new Variable(value.expandDims(axis, times), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.contractDims(axis, times);
			}
		});
	}

	public Variable contractDims(int axis, int times) {
		return new Variable(value.contractDims(axis, times), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.expandDims(axis, times);
			}
		});
	}

	public Variable broadcastTo(int... shape) {
		return new Variable(value.broadcastTo(shape), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				IntArray oldShape = Variable.this.value.shape();
				for (int axis = 0; axis < oldShape.length; axis++) {
					if (oldShape.get(axis) != shape[axis]) {
						forwardGradient = forwardGradient.sumAxis(axis);
					}
				}
				return forwardGradient;
			}
		});
	}

	public Variable broadcastTo(IntArray shape) {
		return new Variable(value.broadcastTo(shape), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				IntArray oldShape = Variable.this.value.shape();
				for (int axis = 0; axis < oldShape.length; axis++) {
					if (oldShape.get(axis) != shape.get(axis)) {
						forwardGradient = forwardGradient.sumAxis(axis);
					}
				}
				return forwardGradient;
			}
		});
	}

	public Variable negative() {
		return new Variable(value.negative(), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.mul(-1);
			}
		});
	}

	public Variable abs() {
		return new Variable(value.abs(), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return value.sign();
			}
		});
	}

	public Variable add(float constant) {
		return new Variable(value.add(constant), new OnesGradientComputation(this));
	}

	public Variable add(Tensor constant) {
		return new Variable(value.add(constant), new OnesGradientComputation(this));
	}

	public Variable add(Variable other) {
		return new Variable(this.value.add(other.value), new OnesGradientComputation(this),
				new OnesGradientComputation(other));
	}

	public Variable sub(float constant) {
		return new Variable(value.sub(constant), new OnesGradientComputation(this));
	}

	public Variable sub(Tensor constant) {
		return new Variable(value.sub(constant), new OnesGradientComputation(this));
	}

	public Variable sub(Variable other) {
		return new Variable(this.value.sub(other.value), new OnesGradientComputation(this), new Computation(other) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.mul(-1);
			}
		});
	}

	public Variable mul(float constant) {
		return new Variable(value.mul(constant), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.mul(constant);
			}
		});
	}

	public Variable mul(Tensor constant) {
		return new Variable(value.mul(constant), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return constant;
			}
		});
	}

	public Variable mul(Variable other) {
		return new Variable(this.value.mul(other.value), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return other.value;
			}
		}, new Computation(other) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value;
			}
		});
	}

	public Variable div(float constant) {
		return new Variable(value.div(constant), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.div(constant);
			}
		});
	}

	public Variable div(Tensor constant) {
		return new Variable(value.div(constant), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return constant.reciprocal();
			}
		});
	}

	public Variable div(Variable other) {
		Tensor resultValue = this.value.div(other.value);
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return other.value.reciprocal();
			}
		}, new Computation(other) {

			@Override
			protected Tensor gradient() {
				return resultValue.div(other.value).negative();
			}
		});
	}

	public Variable reciprocal() {
		Tensor resultValue = value.reciprocal();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.div(Variable.this.value).negative();
			}
		});
	}

	public Variable sqrt() {
		Tensor resultValue = value.sqrt();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.mul(2).reciprocal();
			}
		});
	}

	public Variable exp() {
		Tensor resultValue = value.exp();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue;
			}
		});
	}

	public Variable pow(float exponent) {
		return new Variable(value.pow(exponent), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.pow(exponent - 1).mul(exponent);
			}
		});
	}

	public Variable pow(Tensor exponent) {
		return new Variable(value.pow(exponent), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.pow(exponent.sub(1)).mul(exponent);
			}
		});
	}

	public Variable pow(Variable exponent) {
		Tensor resultValue = this.value.pow(exponent.value);
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.pow(exponent.value.sub(1)).mul(exponent.value);
			}
		}, new Computation(exponent) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.ln().mul(resultValue);
			}
		});
	}

	public Variable ln() {
		return new Variable(value.ln(), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.reciprocal();
			}
		});
	}

	public Variable log(Tensor antilogarithm) {
		Tensor resultValue = value.log(antilogarithm);
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.div(Variable.this.value.ln()).div(Variable.this.value).negative();
			}
		});
	}

	public Variable log(Variable antilogarithm) {
		Tensor resultValue = this.value.log(antilogarithm.value);
		CachedComputation lnThis = new CachedComputation() {

			@Override
			protected Tensor compute() {
				return Variable.this.value.ln();
			}
		};
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.div(lnThis.result()).div(Variable.this.value).negative();
			}
		}, new Computation(antilogarithm) {

			@Override
			protected Tensor gradient() {
				return lnThis.result().mul(antilogarithm.value).reciprocal();
			}
		});
	}

	public Variable tanh() {
		Tensor resultValue = value.tanh();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.pow(2).negative().add(1);
			}
		});
	}

	public Variable relu() {
		Tensor resultValue = value.relu();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue.sign();
			}
		});
	}

	public Variable transpose() {
		return new Variable(this.value.transpose(), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.transpose();
			}
		});
	}

	public Variable transpose(int... permutation) {
		return new Variable(this.value.transpose(permutation), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				if (null == permutation) {
					return forwardGradient.transpose();
				} else {
					int[] backwardPermutation = new int[permutation.length];
					for (int i = 0; i < permutation.length; i++) {
						backwardPermutation[permutation[i]] = i;
					}
					return forwardGradient.transpose(backwardPermutation);
				}
			}
		});
	}

	public Variable dot(Tensor constant) {
		return new Variable(this.value.dot(constant), new RightDotComputation(this, constant));
	}

	public Variable dot(Variable other) {
		return new Variable(this.value.dot(other.value), new RightDotComputation(this, other.value),
				new LeftDotComputation(other, this.value));
	}

	public Tensor value() {
		return value;
	}

	public void value(Tensor value) {
		this.value = value;
	}

	public Tensor gradient() {
		return gradient;
	}

	public void zeroGradient() {
		gradient.zeros();
	}

	public boolean requiresGradient() {
		return requiresGradient;
	}
}
