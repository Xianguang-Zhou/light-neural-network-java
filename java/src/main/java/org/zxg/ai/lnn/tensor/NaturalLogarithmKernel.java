/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor;

import com.aparapi.Kernel;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
class NaturalLogarithmKernel extends Kernel {

	float[] antilogarithm, result;

	NaturalLogarithmKernel(float[] antilogarithm, float[] result) {
		this.antilogarithm = antilogarithm;
		this.result = result;
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		int i = getGlobalId();
		result[i] = log(antilogarithm[i]);
	}
}