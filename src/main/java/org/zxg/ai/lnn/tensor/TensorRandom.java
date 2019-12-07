/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.zxg.ai.lnn.tensor;

import org.zxg.ai.lnn.opencl.Device;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.tensor.kernel.NormalRandomKernel;
import org.zxg.ai.lnn.tensor.kernel.ShuffleIntKernel;
import org.zxg.ai.lnn.tensor.kernel.UniformRandomKernel;

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

	protected long seed() {
		return seed;
	}

	public void seed(long seed) {
		this.seed = (seed ^ MULTIPLIER) & MASK;
	}

	protected void nextSeed() {
		seed = (seed * MULTIPLIER + ADDEND) & MASK;
	}

	public void uniform(Tensor t) {
		uniform(0, 1, t);
	}

	public void uniform(float low, float high, Tensor t) {
		nextSeed();
		t.device().kernel(UniformRandomKernel.class).execute(MASK, MULTIPLIER, ADDEND, seed, low, high, t.flatData());
	}

	public void normal(Tensor t) {
		normal(0, 1, t);
	}

	public void normal(float mean, float standardDeviation, Tensor t) {
		nextSeed();
		t.device().kernel(NormalRandomKernel.class).execute(MASK, MULTIPLIER, ADDEND, seed, mean, standardDeviation,
				t.flatData());
	}

	public void shuffle(Device device, IntArray array) {
		nextSeed();
		device.kernel(ShuffleIntKernel.class).execute(MASK, MULTIPLIER, ADDEND, seed, array);
	}
}
