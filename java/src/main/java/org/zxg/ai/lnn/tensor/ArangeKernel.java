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
class ArangeKernel extends Kernel {

	@Constant
	float[] range_$constant$;
	@Constant
	int[] repeat_$constant$;
	float[] result;

	ArangeKernel(float start, float stop, float step, int repeat, float[] result) {
		this.range_$constant$ = new float[] { start, stop, step };
		this.repeat_$constant$ = new int[] { repeat };
		this.result = result;
	}

	@Override
	public void run() {
		int i = getGlobalId();
		float value = range_$constant$[0] + (i / repeat_$constant$[0]) * range_$constant$[2];
		if (value < range_$constant$[1]) {
			result[i] = value;
		}
	}
}
