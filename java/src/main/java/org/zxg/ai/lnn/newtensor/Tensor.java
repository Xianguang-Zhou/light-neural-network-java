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
import org.zxg.ai.lnn.newtensor.kernel.NegativeKernel;
import org.zxg.ai.lnn.newtensor.kernel.ReciprocalKernel;
import org.zxg.ai.lnn.newtensor.kernel.ReluKernel;
import org.zxg.ai.lnn.newtensor.kernel.SignKernel;
import org.zxg.ai.lnn.opencl.Device;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
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
		precision = DEFAULT_PRECISION;
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
		precision = DEFAULT_PRECISION;
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
		precision = DEFAULT_PRECISION;
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
		precision = DEFAULT_PRECISION;
		this.device = device;
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
		Tensor result = new Tensor(shape);
		kernel(AxisSliceKernel.class).execute(axis, begin, this, result);
		return result;
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
		return new Tensor(shape.clone());
	}

	public final Tensor zerosLike() {
		return like();
	}

	public final Tensor onesLike() {
		Tensor result = like();
		result.ones();
		return result;
	}

	public final Tensor broadcastTo(int... shape) {
		checkSameDim(this.shape, shape);
		for (int i = 0; i < this.shape.length; i++) {
			int length = this.shape.get(i);
			if (length != shape[i] && length != 1) {
				throw new ShapeException();
			}
		}
		Tensor result = new Tensor(shape);
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
		Tensor result = new Tensor(shape);
		kernel(AddKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor add(float value) {
		Tensor result = new Tensor(shape);
		kernel(AddValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor div(Tensor other) {
		checkSameShape(other);
		Tensor result = new Tensor(shape);
		kernel(DivideKernel.class).execute(data, other.data, result.data);
		return result;
	}

	public final Tensor div(float value) {
		Tensor result = new Tensor(shape);
		kernel(DivideValueKernel.class).execute(data, value, result.data);
		return result;
	}

	public final Tensor reciprocal() {
		Tensor result = like();
		kernel(ReciprocalKernel.class).execute(data, result.data);
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
