/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.newtensor;

import org.zxg.ai.lnn.opencl.IntArray;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
final class ShapeInfo {

	public final int size;
	public final IntArray dimSizes;

	public ShapeInfo(int size, IntArray dimSizes) {
		this.size = size;
		this.dimSizes = dimSizes;
	}

	public static ShapeInfo create(int[] shape) {
		int size = 1;
		IntArray dimSizes = new IntArray(shape.length);
		for (int i = shape.length - 1; i >= 0; i--) {
			dimSizes.set(i, size);
			int length = shape[i];
			if (length > 0) {
				size *= length;
			} else {
				throw new ShapeException();
			}
		}
		return new ShapeInfo(size, dimSizes);
	}

	public static ShapeInfo create(IntArray shape) {
		int size = 1;
		IntArray dimSizes = new IntArray(shape.length);
		for (int i = shape.length - 1; i >= 0; i--) {
			dimSizes.set(i, size);
			int length = shape.get(i);
			if (length > 0) {
				size *= length;
			} else {
				throw new ShapeException();
			}
		}
		return new ShapeInfo(size, dimSizes);
	}
}
