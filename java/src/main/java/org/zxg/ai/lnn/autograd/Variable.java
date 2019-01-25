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

	private Record record;
	private Tensor argument;
	private Tensor gradient;
	private Computation[] computations;

	private Variable(Record record, Tensor argument, Computation[] computations) {
		this.record = record;
		this.argument = argument;
		if (null == computations) {
			this.gradient = new Tensor(argument.shape());
		} else {
			this.computations = computations;
		}
	}

	public Variable(Tensor argument) {
		this(Record.current(), argument);
	}

	public Variable(Record record, Tensor argument) {
		this(record, argument, null);
	}

	protected Variable(Tensor argument, Computation... computations) {
		this(computations[0].creator.record, argument, computations);
	}

	private final void backwardInternal(Tensor gradient) {
		if (this.computations != null) {
			for (Computation c : this.computations) {
				if (c.isRecorded) {
					c.creator.backwardInternal(gradient.mul(c.gradient()));
				} else {
					c.creator.backwardInternal(gradient);
				}
			}
		} else {
			this.gradient = this.gradient.add(gradient);
		}
	}

	public final void backward(Tensor gradient) {
		if (null == gradient) {
			backward();
			return;
		}
		if (!record.isClosed()) {
			throw new RecordException();
		}
		backwardInternal(gradient);
	}

	public final void backward() {
		if (!record.isClosed()) {
			throw new RecordException();
		}
		if (this.computations != null) {
			for (Computation c : this.computations) {
				if (c.isRecorded) {
					c.creator.backwardInternal(c.gradient());
				} else {
					Tensor gradient = new Tensor(c.creator.argument.shape());
					gradient.ones();
					c.creator.backwardInternal(gradient);
				}
			}
		} else {
			Tensor gradient = new Tensor(this.gradient.shape());
			gradient.ones();
			backwardInternal(gradient);
		}
	}

	public final Variable negative() {
		return new Variable(argument.negative(), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				Tensor g = new Tensor(Variable.this.argument.shape());
				g.constant(-1);
				return g;
			}
		});
	}

	public final Variable add(Tensor constant) {
		return new Variable(argument.add(constant), new OnesGradientComputation(this));
	}

	public final Variable add(Variable other) {
		return new Variable(this.argument.add(other.argument), new OnesGradientComputation(this),
				new OnesGradientComputation(other));
	}

	public final Variable sub(Tensor constant) {
		return new Variable(argument.sub(constant), new OnesGradientComputation(this));
	}

	public final Variable sub(Variable other) {
		return new Variable(this.argument.sub(other.argument), new OnesGradientComputation(this),
				new Computation(other, record.isRecording()) {

					@Override
					protected Tensor gradient() {
						Tensor g = new Tensor(other.argument.shape());
						g.constant(-1);
						return g;
					}
				});
	}

	public final Variable mul(Tensor constant) {
		return new Variable(argument.mul(constant), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				return constant;
			}
		});
	}

	public final Variable mul(Variable other) {
		boolean isRecorded = record.isRecording();
		return new Variable(this.argument.mul(other.argument), new Computation(this, isRecorded) {

			@Override
			protected Tensor gradient() {
				return other.argument;
			}
		}, new Computation(other, isRecorded) {

			@Override
			protected Tensor gradient() {
				return Variable.this.argument;
			}
		});
	}

	public final Variable div(Tensor constant) {
		return new Variable(argument.div(constant), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				return constant.reciprocal();
			}
		});
	}

	public final Variable div(Variable other) {
		boolean isRecorded = record.isRecording();
		return new Variable(this.argument.div(other.argument), new Computation(this, isRecorded) {

			@Override
			protected Tensor gradient() {
				return other.argument.reciprocal();
			}
		}, new Computation(other, isRecorded) {

			@Override
			protected Tensor gradient() {
				return other.argument.mul(other.argument).reciprocal().negative().mul(Variable.this.argument);
			}
		});
	}

	public final Variable reciprocal() {
		return new Variable(argument.reciprocal(), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				return Variable.this.argument.mul(Variable.this.argument).reciprocal().negative();
			}
		});
	}

	public final Variable sqrt() {
		Tensor resultArgument = argument.sqrt();
		return new Variable(resultArgument, new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(resultArgument.shape());
				two.constant(2);
				return resultArgument.mul(two).reciprocal();
			}
		});
	}

	public final Variable exp() {
		Tensor resultArgument = argument.exp();
		return new Variable(resultArgument, new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				return resultArgument;
			}
		});
	}

	public final Variable pow(Tensor exponent) {
		return new Variable(argument.pow(exponent), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				Tensor one = new Tensor(exponent.shape());
				one.ones();
				return Variable.this.argument.pow(exponent.sub(one)).mul(exponent);
			}
		});
	}

	public final Variable pow(Variable exponent) {
		boolean isRecorded = record.isRecording();
		Tensor resultArgument = this.argument.pow(exponent.argument);
		return new Variable(resultArgument, new Computation(this, isRecorded) {

			@Override
			protected Tensor gradient() {
				Tensor one = new Tensor(exponent.argument.shape());
				one.ones();
				return Variable.this.argument.pow(exponent.argument.sub(one)).mul(exponent.argument);
			}
		}, new Computation(exponent, isRecorded) {

			@Override
			protected Tensor gradient() {
				return Variable.this.argument.ln().mul(resultArgument);
			}
		});
	}

	public final Variable ln() {
		return new Variable(argument.ln(), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				return Variable.this.argument.reciprocal();
			}
		});
	}

	public final Variable log(Tensor antilogarithm) {
		return new Variable(argument.log(antilogarithm), new Computation(this, record.isRecording()) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(Variable.this.argument.shape());
				two.constant(2);
				return Variable.this.argument.ln().pow(two).reciprocal().negative()
						.mul(Variable.this.argument.reciprocal()).mul(antilogarithm.ln());
			}
		});
	}

	public final Variable log(Variable antilogarithm) {
		boolean isRecorded = record.isRecording();
		return new Variable(this.argument.log(antilogarithm.argument), new Computation(this, isRecorded) {

			@Override
			protected Tensor gradient() {
				Tensor two = new Tensor(Variable.this.argument.shape());
				two.constant(2);
				return Variable.this.argument.ln().pow(two).reciprocal().negative()
						.mul(Variable.this.argument.reciprocal()).mul(antilogarithm.argument.ln());
			}
		}, new Computation(antilogarithm, isRecorded) {

			@Override
			protected Tensor gradient() {
				return Variable.this.argument.ln().mul(antilogarithm.argument).reciprocal();
			}
		});
	}

	public final Record record() {
		return record;
	}

	public final Tensor argument() {
		return argument;
	}

	public final Tensor gradient() {
		return gradient;
	}
}
