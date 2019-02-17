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

	private static float DEFAULT_PRECISION = 0.00001f;

	public static void defaultPrecision(float precision) {
		DEFAULT_PRECISION = Math.abs(precision);
	}

	float precision;
	float[] data;
	int[] shape;
	int[] dimSizes;

	public Tensor(Tensor other) {
		precision = other.precision;
		data = new float[other.data.length];
		System.arraycopy(other.data, 0, data, 0, data.length);
		shape = new int[other.shape.length];
		System.arraycopy(other.shape, 0, shape, 0, shape.length);
		dimSizes = new int[other.dimSizes.length];
		System.arraycopy(other.dimSizes, 0, dimSizes, 0, dimSizes.length);
	}

	public Tensor(int... shape) {
		precision = DEFAULT_PRECISION;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new float[info.size];
	}

	public Tensor(float[] data, int... shape) {
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

	public final float[] flatData() {
		return data;
	}

	public final void flatData(float... data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = data;
	}

	public final float scalar() {
		return data[0];
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

	protected final int[] dimSizes() {
		return dimSizes;
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

	public final Tensor slice(int begin, int end) {
		return slice(0, begin, end);
	}

	public final Tensor slice(int axis, int begin, int end) {
		if (axis < 0 || axis >= this.shape.length) {
			throw new IndexOutOfBoundsException();
		}
		if (begin < 0 || begin >= end || end > this.shape[axis]) {
			throw new IndexOutOfBoundsException();
		}
		int[] shape = new int[this.shape.length];
		for (int i = 0; i < shape.length; i++) {
			if (i != axis) {
				shape[i] = this.shape[i];
			} else {
				shape[i] = end - begin;
			}
		}
		Tensor result = new Tensor(shape);
		new AxisSliceKernel(axis, begin, this, result).execute();
		return result;
	}

	public final Tensor slice(int[] begin, int[] end) {
		if (begin.length != this.shape.length || end.length != this.shape.length) {
			throw new DimException();
		}
		int[] shape = new int[this.shape.length];
		for (int i = 0; i < this.shape.length; i++) {
			int elementOfBegin = begin[i];
			int elementOfEnd = end[i];
			if (elementOfBegin < 0 || elementOfBegin >= elementOfEnd || elementOfEnd > this.shape[i]) {
				throw new IndexOutOfBoundsException();
			}
			shape[i] = elementOfEnd - elementOfBegin;
		}
		Tensor result = new Tensor(shape);
		new SliceKernel(begin, this, result).execute();
		return result;
	}

	public final void sliceAssign(int[] begin, Tensor value) {
		if (begin.length != this.shape.length || !sameDim(value)) {
			throw new DimException();
		}
		for (int i = 0; i < this.shape.length; i++) {
			int elementOfBegin = begin[i];
			if (elementOfBegin < 0 || elementOfBegin + value.shape[i] > this.shape[i]) {
				throw new IndexOutOfBoundsException();
			}
		}
		new SliceAssignKernel(begin, value, this).execute();
	}

	public final void setShape(int... shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			float[] data = new float[info.size];
			System.arraycopy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
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

	public final Tensor transpose() {
		return transpose(null);
	}

	public final Tensor transpose(int... permutation) {
		if (this.shape.length < 2) {
			return clone();
		} else {
			if (null == permutation) {
				permutation = new int[this.shape.length];
				for (int i = 0, value = permutation.length - 1; i < permutation.length; i++) {
					permutation[i] = value--;
				}
			} else {
				if (permutation.length != this.shape.length) {
					throw new DimException();
				}
				int[] permutationCheckResults = new int[this.shape.length];
				for (int value : permutation) {
					if (value < 0 || value >= permutationCheckResults.length) {
						throw new IndexOutOfBoundsException();
					}
					permutationCheckResults[value]++;
				}
				for (int value : permutationCheckResults) {
					if (value != 1) {
						throw new ShapeException();
					}
				}
			}
			int[] shape = new int[this.shape.length];
			for (int i = 0; i < shape.length; i++) {
				shape[i] = this.shape[permutation[i]];
			}
			Tensor result = new Tensor(shape);
			new TransposeKernel(permutation, this, result).execute();
			return result;
		}
	}

	public final Tensor broadcastTo(int... shape) {
		checkSameDim(this.shape, shape);
		for (int i = 0; i < this.shape.length; i++) {
			int length = this.shape[i];
			if (length != shape[i] && length != 1) {
				throw new ShapeException();
			}
		}
		Tensor result = new Tensor(shape);
		new BroadcastKernel(this, result).execute();
		return result;
	}

	public final Tensor expandDims(int axis, int times) {
		if (axis < 0 || axis > this.shape.length || times < 1) {
			throw new IndexOutOfBoundsException();
		}
		int[] shape = new int[this.shape.length + times];
		System.arraycopy(this.shape, 0, shape, 0, axis);
		int axisAddTimes = axis + times;
		for (int i = axis; i < axisAddTimes;) {
			shape[i++] = 1;
		}
		System.arraycopy(this.shape, axis, shape, axisAddTimes, this.shape.length - axis);
		return reshape(shape);
	}

	public final Tensor contractDims(int axis, int times) {
		int axisAddTimes = axis + times;
		if (axis < 0 || axisAddTimes > this.shape.length || times < 1) {
			throw new IndexOutOfBoundsException();
		}
		for (int i = axis; i < axisAddTimes;) {
			if (this.shape[i++] != 1) {
				throw new ShapeException();
			}
		}
		int[] shape = new int[this.shape.length - times];
		System.arraycopy(this.shape, 0, shape, 0, axis);
		System.arraycopy(this.shape, axisAddTimes, shape, axis, shape.length - axis);
		return reshape(shape);
	}

	public final void constant(float constant) {
		new ConstantKernel(constant, data).execute();
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
		new ArangeKernel(start, stop, step, repeat, data).execute();
	}

	public final Tensor negative() {
		Tensor result = new Tensor(shape);
		new NegativeKernel(data, result.data).execute();
		return result;
	}

	public final Tensor abs() {
		Tensor result = new Tensor(shape);
		new AbsKernel(data, result.data).execute();
		return result;
	}

	public final Tensor sign() {
		Tensor result = new Tensor(shape);
		new SignKernel(data, result.data).execute();
		return result;
	}

	public final Tensor add(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new AddKernel(data, other.data, result.data).execute();
		return result;
	}

	public final Tensor add(float value) {
		Tensor result = new Tensor(shape);
		new AddValueKernel(data, value, result.data).execute();
		return result;
	}

	public final Tensor sub(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new SubtractKernel(data, other.data, result.data).execute();
		return result;
	}

	public final Tensor sub(float value) {
		Tensor result = new Tensor(shape);
		new SubtractValueKernel(data, value, result.data).execute();
		return result;
	}

	public final Tensor mul(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new MultiplyKernel(data, other.data, result.data).execute();
		return result;
	}

	public final Tensor mul(float value) {
		Tensor result = new Tensor(shape);
		new MultiplyValueKernel(data, value, result.data).execute();
		return result;
	}

	public final Tensor div(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		new DivideKernel(data, other.data, result.data).execute();
		return result;
	}

	public final Tensor div(float value) {
		Tensor result = new Tensor(shape);
		new DivideValueKernel(data, value, result.data).execute();
		return result;
	}

	public final Tensor reciprocal() {
		Tensor result = new Tensor(shape);
		new ReciprocalKernel(data, result.data).execute();
		return result;
	}

	public final Tensor dot(Tensor other) {
		if (this.shape.length == 0) {
			return other.mul(this.data[0]);
		} else if (other.shape.length == 0) {
			return this.mul(other.data[0]);
		} else {
			if (this.shape[this.shape.length - 1] != other.shape[0]) {
				throw new ShapeException();
			}
			if (this.shape.length == 1 && other.shape.length == 1) {
				float[] resultData = new float[this.data.length];
				new MultiplyKernel(this.data, other.data, resultData).execute();
				float resultValue = 0;
				for (float value : resultData) {
					resultValue += value;
				}
				return new Tensor(new float[] { resultValue }, new int[0]);
			} else {
				int[] resultShape = new int[this.shape.length + other.shape.length - 2];
				if (this.shape.length > 1) {
					System.arraycopy(this.shape, 0, resultShape, 0, this.shape.length - 1);
				}
				if (other.shape.length > 1) {
					System.arraycopy(other.shape, 1, resultShape, this.shape.length - 1, other.shape.length - 1);
				}
				Tensor result = new Tensor(resultShape);
				new ProductKernel(this, other, result).execute();
				return result;
			}
		}
	}

	public final Tensor sqrt() {
		Tensor result = new Tensor(shape);
		new SquareRootKernel(data, result.data).execute();
		return result;
	}

	public final Tensor exp() {
		Tensor result = new Tensor(shape);
		new NaturalExponentiationKernel(data, result.data).execute();
		return result;
	}

	public final Tensor pow(Tensor exponent) {
		checkSameShape(exponent);
		Tensor result = new Tensor(shape);
		new PowerKernel(data, exponent.data, result.data).execute();
		return result;
	}

	public final Tensor pow(float exponent) {
		Tensor result = new Tensor(shape);
		new PowerValueKernel(data, exponent, result.data).execute();
		return result;
	}

	public final Tensor ln() {
		Tensor result = new Tensor(shape);
		new NaturalLogarithmKernel(data, result.data).execute();
		return result;
	}

	public final Tensor log(Tensor antilogarithm) {
		checkSameShape(antilogarithm);
		Tensor result = new Tensor(shape);
		new LogarithmKernel(data, antilogarithm.data, result.data).execute();
		return result;
	}

	public final Tensor tanh() {
		Tensor result = new Tensor(shape);
		new TanhKernel(data, result.data).execute();
		return result;
	}

	public final Tensor relu() {
		Tensor result = new Tensor(shape);
		new ReluKernel(data, result.data).execute();
		return result;
	}

	protected float selectPrecision(float otherPrecision) {
		return precision < otherPrecision ? precision : otherPrecision;
	}

	public final boolean lessThan(Tensor other) {
		checkSameShape(other);
		return new LessKernel(selectPrecision(other.precision), this.data, other.data).execute();
	}

	public final boolean lessThanOrEquals(Tensor other) {
		checkSameShape(other);
		return new LessEqualKernel(selectPrecision(other.precision), this.data, other.data).execute();
	}

	public final boolean moreThan(Tensor other) {
		return other.lessThan(this);
	}

	public final boolean moreThanOrEquals(Tensor other) {
		return other.lessThanOrEquals(this);
	}

	public final Tensor sumAxis(int axis) {
		if (1 == this.data.length) {
			return clone();
		}
		int[] shape = new int[this.shape.length];
		System.arraycopy(this.shape, 0, shape, 0, shape.length);
		if (axis < 0 || axis > this.shape.length - 1) {
			throw new IndexOutOfBoundsException();
		}
		shape[axis] = 1;
		Tensor result = new Tensor(shape);
		new SumAxisKernel(axis, this, result).execute();
		return result;
	}

	public final Tensor sum() {
		Tensor result = this;
		for (int axis = 0; axis < shape.length; axis++) {
			if (shape[axis] != 1) {
				result = result.sumAxis(axis);
			}
		}
		if (this == result) {
			result = result.clone();
		}
		result.setShape();
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
		if (!Arrays.equals(shape, other.shape)) {
			return false;
		}
		return new EqualKernel(selectPrecision(other.precision), this.data, other.data).execute();
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
