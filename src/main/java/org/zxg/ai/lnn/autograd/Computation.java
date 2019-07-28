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
public abstract class Computation {

	protected final Variable creator;

	public Computation(Variable creator) {
		this.creator = creator;
	}

	protected Tensor gradient() {
		return null;
	}

	protected Tensor backward(Tensor forwardGradient) {
		return gradient().mul(forwardGradient);
	}
}
