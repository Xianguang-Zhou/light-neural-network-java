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

	public final void backward(Tensor gradient) {
		if (null == gradient) {
			throw new NullPointerException();
		}
		if (!record.isClosed()) {
			throw new RecordException();
		}
		if (this.computations != null) {
			for (Computation c : this.computations) {
				if (c.isRecorded) {
					c.creator.backward(gradient.mul(c.gradient()));
				} else {
					c.creator.backward(gradient);
				}
			}
		} else {
			this.gradient = this.gradient.add(gradient);
		}
	}

	public final void backward() {
		if (!record.isClosed()) {
			throw new RecordException();
		}
		if (this.computations != null) {
			for (Computation c : this.computations) {
				if (c.isRecorded) {
					c.creator.backward(c.gradient());
				} else {
					Tensor gradient = new Tensor(c.creator.argument.shape());
					gradient.ones();
					c.creator.backward(gradient);
				}
			}
		} else {
			Tensor gradient = new Tensor(this.gradient.shape());
			gradient.ones();
			backward(gradient);
		}
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
