/*
 * Copyright (c) 2021, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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

import java.util.Deque;
import java.util.LinkedList;

import org.zxg.ai.lnn.opencl.Device;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.tensor.kernel.integer.EqualsKernel;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class IntTensor implements Cloneable {

	private Device device;
	private IntArray data;
	private IntArray shape;
	private IntArray dimSizes;

	public IntTensor(IntTensor other) {
		device = other.device;
		data = other.data.clone();
		shape = other.shape.clone();
		dimSizes = other.dimSizes.clone();
	}

	public IntTensor(int... shape) {
		this(Tensor.defaultDevice(), shape);
	}

	public IntTensor(Device device, int... shape) {
		this.device = device;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new IntArray(info.size);
	}

	public IntTensor(IntArray shape) {
		this(Tensor.defaultDevice(), shape);
	}

	public IntTensor(Device device, IntArray shape) {
		this.device = device;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		this.dimSizes = info.dimSizes;
		this.data = new IntArray(info.size);
	}

	public IntTensor(int[] data, int... shape) {
		this(Tensor.defaultDevice(), data, shape);
	}

	public IntTensor(Device device, int[] data, int... shape) {
		this.device = device;
		this.shape = new IntArray(shape);
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = new IntArray(data);
	}

	public IntTensor(IntArray data, IntArray shape) {
		this(Tensor.defaultDevice(), data, shape);
	}

	public IntTensor(Device device, IntArray data, IntArray shape) {
		this.device = device;
		this.shape = shape;
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			throw new ShapeException();
		}
		this.dimSizes = info.dimSizes;
		this.data = data;
	}

	public IntTensor create(int... shape) {
		return new IntTensor(device, shape);
	}

	public IntTensor create(IntArray shape) {
		return new IntTensor(device, shape);
	}

	public IntTensor create(int[] data, int... shape) {
		return new IntTensor(device, data, shape);
	}

	public IntTensor create(IntArray data, IntArray shape) {
		return new IntTensor(device, data, shape);
	}

	public Device device() {
		return device;
	}

	public void device(Device device) {
		this.device = device;
	}

	protected <T extends Kernel> T kernel(Class<T> type) {
		return device.kernel(type);
	}

	public IntArray flatData() {
		return data;
	}

	public void flatData(int... data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = new IntArray(data);
	}

	public void flatData(IntArray data) {
		if (data.length != this.data.length) {
			throw new ShapeException();
		}
		this.data = data;
	}

	public void data(Object object) {
		if (object instanceof Number) {
			this.data.set(0, ((Number) object).intValue());
		} else if (object instanceof int[]) {
			this.data.set(0, (int[]) object);
		} else if (object instanceof long[]) {
			int index = 0;
			for (long element : (long[]) object) {
				this.data.set(index++, (int) element);
			}
		} else if (object instanceof Object[]) {
			int index = 0;
			Deque<ArrayIterator> iteratorStack = new LinkedList<>();
			iteratorStack.push(new ArrayIterator((Object[]) object));
			do {
				ArrayIterator arrayIterator = iteratorStack.peek();
				if (arrayIterator.hasNext()) {
					Object objectElement = arrayIterator.next();
					if (objectElement instanceof Number) {
						this.data.set(index++, ((Number) objectElement).intValue());
					} else if (objectElement instanceof int[]) {
						int[] intsElement = (int[]) objectElement;
						this.data.set(index, intsElement);
						index += intsElement.length;
					} else if (objectElement instanceof long[]) {
						for (long value : (long[]) objectElement) {
							this.data.set(index++, (int) value);
						}
					} else if (objectElement instanceof Object[]) {
						iteratorStack.push(new ArrayIterator((Object[]) objectElement));
					}
				} else {
					iteratorStack.pop();
				}
			} while (!iteratorStack.isEmpty());
		}
	}

	public long scalar() {
		return data.get(0);
	}

	public int size() {
		return data.length;
	}

	public IntArray shape() {
		return shape;
	}

	public int ndim() {
		return shape.length;
	}

	public IntArray dimSizes() {
		return dimSizes;
	}

	protected static boolean sameShape(IntArray shape1, IntArray shape2) {
		return null != shape1 && shape1.equals(shape2);
	}

	protected static void checkSameShape(IntArray shape1, IntArray shape2) {
		if (!sameShape(shape1, shape2)) {
			throw new ShapeException();
		}
	}

	public boolean sameShape(IntTensor other) {
		return sameShape(shape, other.shape);
	}

	public void checkSameShape(IntTensor other) {
		checkSameShape(shape, other.shape);
	}

	protected static boolean sameDim(IntArray shape1, IntArray shape2) {
		return shape1.length == shape2.length;
	}

	protected static boolean sameDim(IntArray shape1, int[] shape2) {
		return shape1.length == shape2.length;
	}

	protected static void checkSameDim(IntArray shape1, IntArray shape2) {
		if (!sameDim(shape1, shape2)) {
			throw new DimException();
		}
	}

	protected static void checkSameDim(IntArray shape1, int[] shape2) {
		if (!sameDim(shape1, shape2)) {
			throw new DimException();
		}
	}

	public boolean sameDim(IntTensor other) {
		return sameDim(shape, other.shape);
	}

	public void checkSameDim(IntTensor other) {
		checkSameDim(shape, other.shape);
	}

	private int dataIndex(int... indexes) {
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

	public long get(int... indexes) {
		return data.get(dataIndex(indexes));
	}

	public void set(int value, int... indexes) {
		data.set(dataIndex(indexes), value);
	}

	public Element element(int... indexes) {
		return new Element(dataIndex(indexes));
	}

	public class Element {

		private int index;

		Element(int index) {
			this.index = index;
		}

		public int value() {
			return data.get(index);
		}

		public void value(int value) {
			data.set(index, value);
		}
	}

	public void setShape(int... shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			IntArray data = new IntArray(info.size);
			IntArray.copy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
			this.data = data;
		}
		this.shape = new IntArray(shape);
		this.dimSizes = info.dimSizes;
	}

	public void setShape(IntArray shape) {
		ShapeInfo info = ShapeInfo.create(shape);
		if (info.size != data.length) {
			IntArray data = new IntArray(info.size);
			IntArray.copy(this.data, 0, data, 0, this.data.length > info.size ? info.size : this.data.length);
			this.data = data;
		}
		this.shape = shape;
		this.dimSizes = info.dimSizes;
	}

	public IntTensor reshape(int... shape) {
		IntTensor c = clone();
		c.setShape(shape);
		return c;
	}

	public IntTensor reshape(IntArray shape) {
		IntTensor c = clone();
		c.setShape(shape);
		return c;
	}

	public IntTensor like() {
		return create(shape.clone());
	}

	public IntTensor zerosLike() {
		return like();
	}

	public IntTensor expandDims(int axis, int times) {
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

	public IntTensor contractDims(int axis, int times) {
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

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof IntTensor)) {
			return false;
		}
		IntTensor other = (IntTensor) obj;
		if (!shape.equals(other.shape)) {
			return false;
		}
		return kernel(EqualsKernel.class).execute(data, other.data);
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
	public IntTensor clone() {
		return new IntTensor(this);
	}

	protected void appendDim(final StringBuilder b, final int shapeIndex, final int dataIndex) {
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
