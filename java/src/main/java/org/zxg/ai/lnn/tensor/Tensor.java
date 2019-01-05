/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor;

import java.util.Arrays;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Tensor implements Cloneable {

	float[] data;
	int[] shape;
	int[] dimSizes;

	public Tensor(Tensor other) {
		data = new float[other.data.length];
		System.arraycopy(other.data, 0, data, 0, data.length);
		shape = new int[other.shape.length];
		System.arraycopy(other.shape, 0, shape, 0, shape.length);
		dimSizes = new int[other.dimSizes.length];
		System.arraycopy(other.dimSizes, 0, dimSizes, 0, dimSizes.length);
	}

	public Tensor(int... shape) {
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new float[info.size];
	}

	public Tensor(float[] data, int... shape) {
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = data;
	}

	public final float[] flatData() {
		return data;
	}

	public final void flatData(float... data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = data;
	}

	public final int size() {
		return data.length;
	}

	public final int[] shape() {
		return shape;
	}

	public final int ndim() {
		return shape.length;
	}

	protected static final boolean sameShape(int[] shape1, int[] shape2) {
		return Arrays.equals(shape1, shape2);
	}

	protected static final void checkSameShape(int[] shape1, int[] shape2) {
		if (!sameShape(shape1, shape2)) {
			throw new ShapeException();
		}
	}

	public final boolean sameShape(Tensor other) {
		return sameShape(shape, other.shape);
	}

	public final void checkSameShape(Tensor other) {
		checkSameShape(shape, other.shape);
	}

	protected static final boolean sameDim(int[] shape1, int[] shape2) {
		return shape1.length == shape2.length;
	}

	protected static final void checkSameDim(int[] shape1, int[] shape2) {
		if (!sameDim(shape1, shape2)) {
			throw new DimException();
		}
	}

	public final boolean sameDim(Tensor other) {
		return sameDim(shape, other.shape);
	}

	public final void checkSameDim(Tensor other) {
		checkSameDim(shape, other.shape);
	}

	private final int dataIndex(int... indexes) {
		if (indexes.length != shape.length) {
			throw new DimException();
		}
		int i = 0;
		int dsi = 0;
		for (int index : indexes) {
			if (0 <= index && index < shape[dsi]) {
				i += (index * dimSizes[dsi++]);
			} else {
				throw new IndexOutOfBoundsException();
			}
		}
		return i;
	}

	public final float get(int... indexes) {
		return data[dataIndex(indexes)];
	}

	public final void set(float value, int... indexes) {
		data[dataIndex(indexes)] = value;
	}

	public final void reshape(int... shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			float[] data = new float[info.size];
			System.arraycopy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
			this.data = data;
		}
		this.shape = shape;
		this.dimSizes = info.dimSizes;
	}

	public final void constant(float constant) {
		new ConstantKernel(constant, data).execute(data.length);
	}

	public final void constant(double constant) {
		constant((float) constant);
	}

	public final void ones() {
		constant(1);
	}

	public final void zeros() {
		constant(0);
	}

	public final void arange(float stop) {
		arange(0, stop);
	}

	public final void arange(float start, float stop) {
		arange(start, stop, 1);
	}

	public final void arange(float start, float stop, float step) {
		arange(start, stop, step, 1);
	}

	public final void arange(float start, float stop, float step, int repeat) {
		new ArangeKernel(start, stop, step, repeat, data).execute(data.length);
	}

	public final Tensor add(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new AddKernel(data, other.data, result.data).execute(data.length);
		return result;
	}

	public final Tensor sub(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new SubtractKernel(data, other.data, result.data).execute(data.length);
		return result;
	}

	public final Tensor mul(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new MultiplyKernel(data, other.data, result.data).execute(data.length);
		return result;
	}

	public final Tensor div(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new DivideKernel(data, other.data, result.data).execute(data.length);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof Tensor)) {
			return false;
		}
		Tensor other = (Tensor) obj;
		return Arrays.equals(shape, other.shape) && Arrays.equals(data, other.data);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(data);
		result = prime * result + Arrays.hashCode(shape);
		return result;
	}

	@Override
	public Tensor clone() {
		return new Tensor(this);
	}

	protected final void appendDim(final StringBuilder b, final int shapeIndex, final int dataIndex) {
		b.append('[');
		final int dimLength = shape[shapeIndex];
		if (shapeIndex != shape.length - 1) {
			final int dimSize = dimSizes[shapeIndex];
			for (int i = 0; i < dimLength; i++) {
				if (i != 0) {
					b.append(',');
					for (int j = shape.length - 1 - shapeIndex; j > 0; j--) {
						b.append('\n');
					}
					for (int j = 0; j <= shapeIndex; j++) {
						b.append(' ');
					}
				}
				appendDim(b, shapeIndex + 1, dataIndex + i * dimSize);
			}
		} else {
			for (int i = 0; i < dimLength; i++) {
				if (i != 0) {
					b.append(", ");
				}
				b.append(data[dataIndex + i]);
			}
		}
		b.append(']');
	}

	@Override
	public String toString() {
		StringBuilder b = new StringBuilder();
		if (shape.length == 0) {
			b.append(data[0]);
		} else {
			appendDim(b, 0, 0);
		}
		return b.toString();
	}
}
