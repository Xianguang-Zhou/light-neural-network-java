/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.autograd;

import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Variable {

	private Tensor value;
	private Tensor gradient;
	private Computation[] computations;

	public Variable(Tensor value) {
		this.value = value;
		this.gradient = new Tensor(value.shape());
	}

	protected Variable(Tensor value, Computation... computations) {
		this.value = value;
		this.computations = computations;
	}

	private final void backwardInternal(Tensor gradient) {
		if (this.computations != null) {
			for (Computation c : this.computations) {
				c.creator.backwardInternal(c.backward(gradient));
			}
		} else {
			this.gradient = this.gradient.add(gradient);
		}
	}

	public final void backward(Tensor gradient) {
		if (null == gradient) {
			gradient = new Tensor(this.value.shape());
			gradient.ones();
		} else {
			value.checkSameShape(gradient);
		}
		backwardInternal(gradient);
	}

	public final void backward() {
		backward(null);
	}

	public final Variable negative() {
		return new Variable(value.negative(), new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor g = new Tensor(Variable.this.value.shape());
				g.constant(-1);
				return g;
			}
		});
	}

	public final Variable add(Tensor constant) {
		return new Variable(value.add(constant), new OnesGradientComputation(this));
	}

	public final Variable add(Variable other) {
		return new Variable(this.value.add(other.value), new OnesGradientComputation(this),
				new OnesGradientComputation(other));
	}

	public final Variable sub(Tensor constant) {
		return new Variable(value.sub(constant), new OnesGradientComputation(this));
	}

	public final Variable sub(Variable other) {
		return new Variable(this.value.sub(other.value), new OnesGradientComputation(this), new Computation(other) {

			@Override
			protected Tensor gradient() {
				Tensor g = new Tensor(other.value.shape());
				g.constant(-1);
				return g;
			}
		});
	}

	public final Variable mul(Tensor constant) {
		return new Variable(value.mul(constant), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return constant;
			}
		});
	}

	public final Variable mul(Variable other) {
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

	public final Variable div(Tensor constant) {
		return new Variable(value.div(constant), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return constant.reciprocal();
			}
		});
	}

	public final Variable div(Variable other) {
		return new Variable(this.value.div(other.value), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return other.value.reciprocal();
			}
		}, new Computation(other) {

			@Override
			protected Tensor gradient() {
				return other.value.mul(other.value).reciprocal().negative().mul(Variable.this.value);
			}
		});
	}

	public final Variable reciprocal() {
		return new Variable(value.reciprocal(), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.mul(Variable.this.value).reciprocal().negative();
			}
		});
	}

	public final Variable sqrt() {
		Tensor resultValue = value.sqrt();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(resultValue.shape());
				two.constant(2);
				return resultValue.mul(two).reciprocal();
			}
		});
	}

	public final Variable exp() {
		Tensor resultValue = value.exp();
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				return resultValue;
			}
		});
	}

	public final Variable pow(Tensor exponent) {
		return new Variable(value.pow(exponent), new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor one = new Tensor(exponent.shape());
				one.ones();
				return Variable.this.value.pow(exponent.sub(one)).mul(exponent);
			}
		});
	}

	public final Variable pow(Variable exponent) {
		Tensor resultValue = this.value.pow(exponent.value);
		return new Variable(resultValue, new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor one = new Tensor(exponent.value.shape());
				one.ones();
				return Variable.this.value.pow(exponent.value.sub(one)).mul(exponent.value);
			}
		}, new Computation(exponent) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.ln().mul(resultValue);
			}
		});
	}

	public final Variable ln() {
		return new Variable(value.ln(), new Computation(this) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.reciprocal();
			}
		});
	}

	public final Variable log(Tensor antilogarithm) {
		return new Variable(value.log(antilogarithm), new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(Variable.this.value.shape());
				two.constant(2);
				return Variable.this.value.ln().pow(two).reciprocal().negative().mul(Variable.this.value.reciprocal())
						.mul(antilogarithm.ln());
			}
		});
	}

	public final Variable log(Variable antilogarithm) {
		return new Variable(this.value.log(antilogarithm.value), new Computation(this) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(Variable.this.value.shape());
				two.constant(2);
				return Variable.this.value.ln().pow(two).reciprocal().negative().mul(Variable.this.value.reciprocal())
						.mul(antilogarithm.value.ln());
			}
		}, new Computation(antilogarithm) {

			@Override
			protected Tensor gradient() {
				return Variable.this.value.ln().mul(antilogarithm.value).reciprocal();
			}
		});
	}

	public final Variable transpose() {
		return new Variable(this.value.transpose(), new Computation(this) {

			@Override
			protected Tensor backward(Tensor forwardGradient) {
				return forwardGradient.transpose();
			}
		});
	}

	public final Variable transpose(int... permutation) {
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

	public final Variable dot(Tensor constant) {
		return new Variable(this.value.dot(constant), new RightDotComputation(this, constant));
	}

	public final Variable dot(Variable other) {
		return new Variable(this.value.dot(other.value), new RightDotComputation(this, other.value),
				new LeftDotComputation(other, this.value));
	}

	public final Tensor value() {
		return value;
	}

	public final Tensor gradient() {
		return gradient;
	}
}
