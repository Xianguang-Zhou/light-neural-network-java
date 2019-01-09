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
class UniformRandomKernel extends Kernel {

	@Constant
	long[] seed_$constant$;
	@Constant
	float[] interval_$constant$;
	float[] result;

	UniformRandomKernel(long seed, float low, float high, float[] result) {
		this.seed_$constant$ = new long[] { seed };
		this.interval_$constant$ = new float[] { low, high - low };
		this.result = result;
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid0 = getGlobalId();
		long seed = seed_$constant$[0] * (gid0 + 1);
		seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1);
		seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
		final int next24 = (int) (seed >>> 24);
		final float nextFloat = next24 / ((float) (1 << 24));
		result[gid0] = nextFloat * interval_$constant$[1] + interval_$constant$[0];
	}
}
