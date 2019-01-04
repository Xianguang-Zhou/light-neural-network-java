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
final class ShapeInfo {

	public final int size;
	public final int[] dimSizes;

	public ShapeInfo(int size, int[] dimSizes) {
		this.size = size;
		this.dimSizes = dimSizes;
	}

	public static ShapeInfo create(int[] shape) {
		int size = 1;
		int[] dimSizes = new int[shape.length];
		for (int i = shape.length - 1; i >= 0; i--) {
			dimSizes[i] = size;
			int length = shape[i];
			if (length > 0) {
				size *= length;
			} else {
				throw new ShapeException();
			}
		}
		return new ShapeInfo(size, dimSizes);
	}
}
