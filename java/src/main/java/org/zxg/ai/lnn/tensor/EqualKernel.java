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
class EqualKernel extends Kernel {

	@Constant
	float[] precision_$constant$;
	float[] left, right;
	int[] result = new int[] { 0 };

	EqualKernel(float precision, float[] left, float[] right) {
		this.precision_$constant$ = new float[] { precision };
		this.left = left;
		this.right = right;
	}

	boolean execute() {
		execute(left.length);
		dispose();
		return result[0] == 0;
	}

	@Override
	public void run() {
		int i = getGlobalId();
		if (abs(left[i] - right[i]) > precision_$constant$[0]) {
			result[0] = 1;
		}
	}
}
