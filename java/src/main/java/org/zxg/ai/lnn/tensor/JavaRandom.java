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
public class JavaRandom {

	private long seed;

	public JavaRandom() {
		this(0);
	}

	public JavaRandom(long seed) {
		seed(seed);
	}

	public final void seed(long seed) {
		this.seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1);
	}

	public final void next(Tensor t) {
		new JavaRandomKernel(seed, t.data).execute();
		seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	}
}
