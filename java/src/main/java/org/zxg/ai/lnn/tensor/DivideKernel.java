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
class DivideKernel extends Kernel {

	float[] left, right, result;

	DivideKernel(float[] left, float[] right, float[] result) {
		this.left = left;
		this.right = right;
		this.result = result;
	}

	void execute() {
		execute(left.length);
		dispose();
	}

	@Override
	public void run() {
		int i = getGlobalId();
		result[i] = left[i] / right[i];
	}
}
