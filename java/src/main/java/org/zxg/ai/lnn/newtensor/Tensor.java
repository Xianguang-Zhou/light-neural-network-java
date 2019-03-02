/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.newtensor;

import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.IntArray;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Tensor implements Cloneable {

	private static float DEFAULT_PRECISION = 0.00001f;

	public static void defaultPrecision(float precision) {
		DEFAULT_PRECISION = Math.abs(precision);
	}

	float precision;
	FloatArray data;
	IntArray shape;
	IntArray dimSizes;

	public Tensor(Tensor other) {
		precision = other.precision;
		data = other.data.clone();
		shape = other.shape.clone();
		dimSizes = other.dimSizes.clone();
	}

	public Tensor(int... shape) {
		precision = DEFAULT_PRECISION;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(info.size);
	}

	public Tensor(IntArray shape) {
		precision = DEFAULT_PRECISION;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(info.size);
	}

	public Tensor(float[] data, int... shape) {
		precision = DEFAULT_PRECISION;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(data);
	}

	public Tensor(FloatArray data, IntArray shape) {
		precision = DEFAULT_PRECISION;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = data;
	}

	public final float precision() {
		return precision;
	}

	public final void precision(float precision) {
		this.precision = Math.abs(precision);
	}

	public final FloatArray flatData() {
		return data;
	}

	public final void flatData(float... data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = new FloatArray(data);
	}

	public final void flatData(FloatArray data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = data;
	}

	public final float scalar() {
		return data.get(0);
	}

	public final int size() {
		return data.length;
	}

	public final IntArray shape() {
		return shape;
	}

	public final int ndim() {
		return shape.length;
	}

	protected final IntArray dimSizes() {
		return dimSizes;
	}

	protected static final boolean sameShape(IntArray shape1, IntArray shape2) {
		return null != shape1 && shape1.equals(shape2);
	}

	protected static final void checkSameShape(IntArray shape1, IntArray shape2) {
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

	protected static final boolean sameDim(IntArray shape1, IntArray shape2) {
		return shape1.length == shape2.length;
	}

	protected static final void checkSameDim(IntArray shape1, IntArray shape2) {
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
			if (0 <= index && index < shape.get(dsi)) {
				i += (index * dimSizes.get(dsi++));
			} else {
				throw new IndexOutOfBoundsException();
			}
		}
		return i;
	}

	public final float get(int... indexes) {
		return data.get(dataIndex(indexes));
	}

	public final void set(float value, int... indexes) {
		data.set(dataIndex(indexes), value);
	}

	public final void setShape(int... shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			FloatArray data = new FloatArray(info.size);
			FloatArray.copy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
			this.data = data;
		}
		this.shape = new IntArray(shape);
		this.dimSizes = info.dimSizes;
	}

	public final void setShape(IntArray shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			FloatArray data = new FloatArray(info.size);
			FloatArray.copy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
			this.data = data;
		}
		this.shape = shape;
		this.dimSizes = info.dimSizes;
	}

	public final Tensor reshape(int... shape) {
		Tensor c = clone();
		c.setShape(shape);
		return c;
	}

	public final Tensor reshape(IntArray shape) {
		Tensor c = clone();
		c.setShape(shape);
		return c;
	}

	public final Tensor zerosLike() {
		return new Tensor(shape.clone());
	}

	public final Tensor expandDims(int axis, int times) {
		if (axis < 0 || axis > this.shape.length || times < 1) {
			throw new IndexOutOfBoundsException();
		}
		IntArray shape = new IntArray(this.shape.length + times);
		IntArray.copy(this.shape, 0, shape, 0, axis);
		int axisAddTimes = axis + times;
		for (int i = axis; i < axisAddTimes;) {
			shape.set(i++, 1);
		}
		IntArray.copy(this.shape, axis, shape, axisAddTimes, this.shape.length - axis);
		return reshape(shape);
	}

	public final Tensor contractDims(int axis, int times) {
		int axisAddTimes = axis + times;
		if (axis < 0 || axisAddTimes > this.shape.length || times < 1) {
			throw new IndexOutOfBoundsException();
		}
		for (int i = axis; i < axisAddTimes;) {
			if (this.shape.get(i++) != 1) {
				throw new ShapeException();
			}
		}
		IntArray shape = new IntArray(this.shape.length - times);
		IntArray.copy(this.shape, 0, shape, 0, axis);
		IntArray.copy(this.shape, axisAddTimes, shape, axis, shape.length - axis);
		return reshape(shape);
	}

	protected float selectPrecision(float otherPrecision) {
		return precision < otherPrecision ? precision : otherPrecision;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + data.hashCode();
		result = prime * result + shape.hashCode();
		return result;
	}

	@Override
	public Tensor clone() {
		return new Tensor(this);
	}

	protected final void appendDim(final StringBuilder b, final int shapeIndex, final int dataIndex) {
		b.append('[');
		final int dimLength = shape.get(shapeIndex);
		if (shapeIndex != shape.length - 1) {
			final int dimSize = dimSizes.get(shapeIndex);
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
				b.append(data.get(dataIndex + i));
			}
		}
		b.append(']');
	}

	@Override
	public String toString() {
		StringBuilder b = new StringBuilder();
		if (shape.length == 0) {
			b.append(data.get(0));
		} else {
			appendDim(b, 0, 0);
		}
		return b.toString();
	}
}
