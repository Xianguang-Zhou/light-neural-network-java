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
