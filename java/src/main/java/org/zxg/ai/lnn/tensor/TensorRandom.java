/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class TensorRandom {

	protected static final long MASK = (1L << 48) - 1;
	protected static final long MULTIPLIER = 0x5DEECE66DL;
	protected static final long ADDEND = 0xBL;

	private long seed;

	public TensorRandom() {
		this(0);
	}

	public TensorRandom(long seed) {
		seed(seed);
	}

	protected final long seed() {
		return seed;
	}

	public final void seed(long seed) {
		this.seed = (seed ^ MULTIPLIER) & MASK;
	}

	protected final void nextSeed() {
		seed = (seed * MULTIPLIER + ADDEND) & MASK;
	}

	public final void uniform(Tensor t) {
		uniform(0, 1, t);
	}

	public final void uniform(float low, float high, Tensor t) {
		nextSeed();
		new UniformRandomKernel(seed, low, high, t.data).execute();
	}

	public final void normal(Tensor t) {
		normal(0, 1, t);
	}

	public final void normal(float mean, float standardDeviation, Tensor t) {
		nextSeed();
		new NormalRandomKernel(seed, mean, standardDeviation, t.data).execute();
	}
}
