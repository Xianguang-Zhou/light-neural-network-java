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
class ConstantKernel extends Kernel {

	@Constant
	float[] constant_$constant$;
	float[] result;

	ConstantKernel(float constant, float[] result) {
		this.constant_$constant$ = new float[] { constant };
		this.result = result;
	}

	@Override
	public void run() {
		int i = getGlobalId();
		result[i] = constant_$constant$[0];
	}
}
