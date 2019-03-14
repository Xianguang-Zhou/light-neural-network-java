/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.newtensor;

import org.zxg.ai.lnn.newtensor.kernel.AbsKernel;
import org.zxg.ai.lnn.newtensor.kernel.AddKernel;
import org.zxg.ai.lnn.newtensor.kernel.AddValueKernel;
import org.zxg.ai.lnn.newtensor.kernel.ArangeKernel;
import org.zxg.ai.lnn.newtensor.kernel.AxisSliceKernel;
import org.zxg.ai.lnn.newtensor.kernel.BroadcastKernel;
import org.zxg.ai.lnn.newtensor.kernel.ConstantKernel;
import org.zxg.ai.lnn.newtensor.kernel.DivideKernel;
import org.zxg.ai.lnn.newtensor.kernel.DivideValueKernel;
import org.zxg.ai.lnn.newtensor.kernel.EqualKernel;
import org.zxg.ai.lnn.newtensor.kernel.LogarithmKernel;
import org.zxg.ai.lnn.newtensor.kernel.MultiplyKernel;
import org.zxg.ai.lnn.newtensor.kernel.MultiplyValueKernel;
import org.zxg.ai.lnn.newtensor.kernel.NaturalExponentiationKernel;
import org.zxg.ai.lnn.newtensor.kernel.NaturalLogarithmKernel;
import org.zxg.ai.lnn.newtensor.kernel.NegativeKernel;
import org.zxg.ai.lnn.newtensor.kernel.PowerKernel;
import org.zxg.ai.lnn.newtensor.kernel.PowerValueKernel;
import org.zxg.ai.lnn.newtensor.kernel.ProductKernel;
import org.zxg.ai.lnn.newtensor.kernel.ReciprocalKernel;
import org.zxg.ai.lnn.newtensor.kernel.ReluKernel;
import org.zxg.ai.lnn.newtensor.kernel.SignKernel;
import org.zxg.ai.lnn.newtensor.kernel.SliceAssignKernel;
import org.zxg.ai.lnn.newtensor.kernel.SliceKernel;
import org.zxg.ai.lnn.newtensor.kernel.SquareRootKernel;
import org.zxg.ai.lnn.newtensor.kernel.SubtractKernel;
import org.zxg.ai.lnn.newtensor.kernel.SubtractValueKernel;
import org.zxg.ai.lnn.newtensor.kernel.SumAxisKernel;
import org.zxg.ai.lnn.newtensor.kernel.TanhKernel;
import org.zxg.ai.lnn.newtensor.kernel.TransposeKernel;
import org.zxg.ai.lnn.newtensor.kernel.VectorProductKernel;
import org.zxg.ai.lnn.opencl.Device;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.tensor.DimException;
import org.zxg.ai.lnn.tensor.ShapeException;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Tensor implements Cloneable {

	private static float DEFAULT_PRECISION = 0.00001f;
	private static Device DEFAULT_DEVICE;

	public static void defaultPrecision(float precision) {
		DEFAULT_PRECISION = Math.abs(precision);
	}

	public static void defaultDevice(Device device) {
		DEFAULT_DEVICE = device;
	}

	public static Device defaultDevice() {
		if (null == DEFAULT_DEVICE) {
			DEFAULT_DEVICE = Device.defaultDevice();
		}
		return DEFAULT_DEVICE;
	}

	private float precision;
	private Device device;
	private FloatArray data;
	private IntArray shape;
	private IntArray dimSizes;

	public Tensor(Tensor other) {
		precision = other.precision;
		device = other.device;
		data = other.data.clone();
		shape = other.shape.clone();
		dimSizes = other.dimSizes.clone();
	}

	public Tensor(int... shape) {
		this(defaultDevice(), shape);
	}

	public Tensor(Device device, int... shape) {
		this(DEFAULT_PRECISION, device, shape);
	}

	public Tensor(float precision, Device device, int... shape) {
		this.precision = precision;
		this.device = device;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(info.size);
	}

	public Tensor(IntArray shape) {
		this(defaultDevice(), shape);
	}

	public Tensor(Device device, IntArray shape) {
		this(DEFAULT_PRECISION, device, shape);
	}

	public Tensor(float precision, Device device, IntArray shape) {
		this.precision = precision;
		this.device = device;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(info.size);
	}

	public Tensor(float[] data, int... shape) {
		this(defaultDevice(), data, shape);
	}

	public Tensor(Device device, float[] data, int... shape) {
		this(DEFAULT_PRECISION, device, data, shape);
	}

	public Tensor(float precision, Device device, float[] data, int... shape) {
		this.precision = precision;
		this.device = device;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = new FloatArray(data);
	}

	public Tensor(FloatArray data, IntArray shape) {
		this(defaultDevice(), data, shape);
	}

	public Tensor(Device device, FloatArray data, IntArray shape) {
		this(DEFAULT_PRECISION, device, data, shape);
	}

	public Tensor(float precision, Device device, FloatArray data, IntArray shape) {
		this.precision = precision;
		this.device = device;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = data;
	}

	public final Tensor create(int... shape) {
		return new Tensor(precision, device, shape);
	}

	public final Tensor create(IntArray shape) {
		return new Tensor(precision, device, shape);
	}

	public final Tensor create(float[] data, int... shape) {
		return new Tensor(precision, device, data, shape);
	}

	public final Tensor create(FloatArray data, IntArray shape) {
		return new Tensor(precision, device, data, shape);
	}

	public final float precision() {
		return precision;
	}

	public final void precision(float precision) {
		this.precision = Math.abs(precision);
	}

	public final Device device() {
		return device;
	}

	public final void device(Device device) {
		this.device = device;
	}

	protected final <T extends Kernel> T kernel(Class<T> type) {
		return device.kernel(type);
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

	public final IntArray dimSizes() {
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

	protected static final boolean sameDim(IntArray shape1, int[] shape2) {
		return shape1.length == shape2.length;
	}

	protected static final void checkSameDim(IntArray shape1, IntArray shape2) {
		if (!sameDim(shape1, shape2)) {
			throw new DimException();
		}
	}

	protected static final void checkSameDim(IntArray shape1, int[] shape2) {
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

	public final Tensor slice(int begin, int end) {
		return slice(0, begin, end);
	}

	public final Tensor slice(int axis, int begin, int end) {
		if (axis < 0 || axis >= this.shape.length) {
			throw new IndexOutOfBoundsException();
		}
		if (begin < 0 || begin >= end || end > this.shape.get(axis)) {
			throw new IndexOutOfBoundsException();
		}
		int[] shape = new int[this.shape.length];
		for (int i = 0; i < shape.length; i++) {
			if (i != axis) {
				shape[i] = this.shape.get(i);
			} else {
				shape[i] = end - begin;
			}
		}
		Tensor result = create(shape);
		kernel(AxisSliceKernel.class).execute(axis, begin, this, result);
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
			if (elementOfBegin < 0 || elementOfBegin >= elementOfEnd || elementOfEnd > this.shape.get(i)) {
				throw new IndexOutOfBoundsException();
			}
			shape[i] = elementOfEnd - elementOfBegin;
		}
		Tensor result = create(shape);
		kernel(SliceKernel.class).execute(new IntArray(begin), this, result);
		return result;
	}

	public final void sliceAssign(int[] begin, Tensor value) {
		if (begin.length != this.shape.length || !sameDim(value)) {
			throw new DimException();
		}
		for (int i = 0; i < this.shape.length; i++) {
			int elementOfBegin = begin[i];
			if (elementOfBegin < 0 || elementOfBegin + value.shape.get(i) > this.shape.get(i)) {
				throw new IndexOutOfBoundsException();
			}
		}
		kernel(SliceAssignKernel.class).execute(new IntArray(begin), value, this);
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

	public final Tensor like() {
		return create(shape.clone());
	}

	public final Tensor zerosLike() {
		return like();
	}

	public final Tensor onesLike() {
		Tensor result = like();
		result.ones();
		return result;
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
				shape[i] = this.shape.get(permutation[i]);
			}
			Tensor result = create(shape);
			kernel(TransposeKernel.class).execute(new IntArray(permutation), this, result);
			return result;
		}
	}

	public final Tensor broadcastTo(int... shape) {
		checkSameDim(this.shape, shape);
		for (int i = 0; i < this.shape.length; i++) {
			int length = this.shape.get(i);
			if (length != shape[i] && length != 1) {
				throw new ShapeException();
			}
		}
		Tensor result = create(shape);
		kernel(BroadcastKernel.class).execute(this, result);
		return result;
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

	public final void constant(float constant) {
		kernel(ConstantKernel.class).execute(constant, data);
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
		kernel(ArangeKernel.class).execute(start, stop, step, repeat, data);
	}

	public final Tensor negative() {
		Tensor result = like();
		kernel(NegativeKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor abs() {
		Tensor result = like();
		kernel(AbsKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor sign() {
		Tensor result = like();
		kernel(SignKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor add(Tensor other) {
		checkSameShape(other);
		Tensor result = like();
		kernel(AddKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor add(float value) {
		Tensor result = like();
		kernel(AddValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor sub(Tensor other) {
		checkSameShape(other);
		Tensor result = like();
		kernel(SubtractKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor sub(float value) {
		Tensor result = like();
		kernel(SubtractValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor mul(Tensor other) {
		checkSameShape(other);
		Tensor result = like();
		kernel(MultiplyKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor mul(float value) {
		Tensor result = like();
		kernel(MultiplyValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor div(Tensor other) {
		checkSameShape(other);
		Tensor result = like();
		kernel(DivideKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor div(float value) {
		Tensor result = like();
		kernel(DivideValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor reciprocal() {
		Tensor result = like();
		kernel(ReciprocalKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor dot(Tensor other) {
		if (this.shape.length == 0) {
			return other.mul(this.data.get(0));
		} else if (other.shape.length == 0) {
			return this.mul(other.data.get(0));
		} else {
			if (this.shape.get(this.shape.length - 1) != other.shape.get(0)) {
				throw new ShapeException();
			}
			if (this.shape.length == 1 && other.shape.length == 1) {
				Tensor result = create(new IntArray(0));
				kernel(VectorProductKernel.class).execute(this, other, result);
				return result;
			} else {
				IntArray resultShape = new IntArray(this.shape.length + other.shape.length - 2);
				if (this.shape.length > 1) {
					IntArray.copy(this.shape, 0, resultShape, 0, this.shape.length - 1);
				}
				if (other.shape.length > 1) {
					IntArray.copy(other.shape, 1, resultShape, this.shape.length - 1, other.shape.length - 1);
				}
				Tensor result = create(resultShape);
				kernel(ProductKernel.class).execute(this, other, result);
				return result;
			}
		}
	}

	public final Tensor sqrt() {
		Tensor result = like();
		kernel(SquareRootKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor exp() {
		Tensor result = like();
		kernel(NaturalExponentiationKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor pow(Tensor exponent) {
		checkSameShape(exponent);
		Tensor result = like();
		kernel(PowerKernel.class).execute(data, exponent.data, result.data);
		return result;
	}

	public final Tensor pow(float exponent) {
		Tensor result = like();
		kernel(PowerValueKernel.class).execute(data, exponent, result.data);
		return result;
	}

	public final Tensor ln() {
		Tensor result = like();
		kernel(NaturalLogarithmKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor log(Tensor antilogarithm) {
		checkSameShape(antilogarithm);
		Tensor result = like();
		kernel(LogarithmKernel.class).execute(data, antilogarithm.data, result.data);
		return result;
	}

	public final Tensor tanh() {
		Tensor result = like();
		kernel(TanhKernel.class).execute(data, result.data);
		return result;
	}

	public final Tensor relu() {
		Tensor result = like();
		kernel(ReluKernel.class).execute(data, result.data);
		return result;
	}

	protected float selectPrecision(float otherPrecision) {
		return precision < otherPrecision ? precision : otherPrecision;
	}

	public final Tensor sumAxis(int axis) {
		if (1 == this.data.length) {
			return clone();
		}
		IntArray shape = new IntArray(this.shape.length);
		IntArray.copy(this.shape, 0, shape, 0, shape.length);
		if (axis < 0 || axis > this.shape.length - 1) {
			throw new IndexOutOfBoundsException();
		}
		shape.set(axis, 1);
		Tensor result = create(shape);
		kernel(SumAxisKernel.class).execute(axis, this, result);
		return result;
	}

	public final Tensor sum() {
		Tensor result = this;
		for (int axis = 0; axis < shape.length; axis++) {
			if (shape.get(axis) != 1) {
				result = result.sumAxis(axis);
			}
		}
		if (this == result) {
			result = result.clone();
		}
		result.setShape(new IntArray(0));
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
		if (!shape.equals(other.shape)) {
			return false;
		}
		return kernel(EqualKernel.class).execute(selectPrecision(other.precision), this.data, other.data);
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

	public void print() {
		System.out.println(toString());
	}
}
