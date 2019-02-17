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
class NormalRandomKernel extends Kernel {

	private static final float TWO_PI = (float) (2 * Math.PI);

	@Constant
	long[] seed_$constant$;
	@Constant
	float[] parameters_$constant$;
	float[] result;

	NormalRandomKernel(long seed, float mean, float standardDeviation, float[] result) {
		this.seed_$constant$ = new long[] { seed };
		this.parameters_$constant$ = new float[] { mean, standardDeviation };
		this.result = result;
	}

	void execute() {
		execute(result.length);
		dispose();
	}

	@Override
	public void run() {
		final int gid0 = getGlobalId();
		final long seed = seed_$constant$[0];
		if ((gid0 % 2) == 0) {
			final float u1 = nextFloat(seed, gid0);
			final float u2 = nextFloat(seed, gid0 + 1);
			final float z0 = sqrt(-2 * log(u1)) * cos(TWO_PI * u2);
			result[gid0] = z0 * parameters_$constant$[1] + parameters_$constant$[0];
		} else {
			final float u1 = nextFloat(seed, gid0 - 1);
			final float u2 = nextFloat(seed, gid0);
			final float z1 = sqrt(-2 * log(u1)) * sin(TWO_PI * u2);
			result[gid0] = z1 * parameters_$constant$[1] + parameters_$constant$[0];
		}
	}

	private static float nextFloat(long seed, int index) {
		seed = seed * (index + 1);
		seed = (seed ^ TensorRandom.MULTIPLIER) & TensorRandom.MASK;
		seed = (seed * TensorRandom.MULTIPLIER + TensorRandom.ADDEND) & TensorRandom.MASK;
		final int next24 = (int) (seed >>> 24);
		return next24 / ((float) (1 << 24));
	}
}
