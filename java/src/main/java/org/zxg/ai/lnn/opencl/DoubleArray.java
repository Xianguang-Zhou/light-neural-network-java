/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.nio.DoubleBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class DoubleArray extends BufferArray {

	private final DoubleBuffer buffer;
	public final int length;

	public static void copy(DoubleArray src, int srcPos, DoubleArray dest, int destPos, int length) {
		final DoubleBuffer srcBuffer = src.buffer;
		final DoubleBuffer destBuffer = dest.buffer;
		final int originalSrcPosition = srcBuffer.position();
		final int originalSrcLimit = srcBuffer.limit();
		final int originalDestPosition = destBuffer.position();
		try {
			srcBuffer.position(srcPos);
			srcBuffer.limit(srcPos + length);
			destBuffer.position(destPos);
			destBuffer.put(srcBuffer);
		} finally {
			srcBuffer.position(originalSrcPosition);
			srcBuffer.limit(originalSrcLimit);
			destBuffer.position(originalDestPosition);
		}
	}

	public static DoubleArray copyOfRange(DoubleArray original, int from, int to) {
		int newLength = to - from;
		if (newLength < 0) {
			throw new IllegalArgumentException(from + " > " + to);
		}
		DoubleArray newArray = new DoubleArray(newLength);
		copy(original, from, newArray, 0, Math.min(original.length - from, newLength));
		return newArray;
	}

	public DoubleArray(int length) {
		this.buffer = Buffers.newDirectDoubleBuffer(length);
		this.length = length;
	}

	public DoubleArray(double[] elements) {
		this.buffer = Buffers.newDirectDoubleBuffer(elements);
		this.length = elements.length;
	}

	public DoubleArray(DoubleArray other) {
		this.buffer = Buffers.copyDoubleBuffer(other.buffer);
		this.length = other.length;
	}

	public DoubleArray(DoubleBuffer buffer) {
		this.buffer = Buffers.copyDoubleBuffer(buffer);
		this.length = this.buffer.capacity() / Buffers.SIZEOF_DOUBLE;
	}

	public double get(int index) {
		return buffer.get(index);
	}

	public void set(int index, double value) {
		buffer.put(index, value);
	}

	public void get(int begin, double[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.get(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
		}
	}

	public double[] get(int begin, int end) {
		double[] elements = new double[end - begin];
		get(begin, elements, 0, elements.length);
		return elements;
	}

	public double[] get() {
		return get(0, length);
	}

	public void set(int begin, double[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.put(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
		}
	}

	public void set(int begin, double[] elements, int offset) {
		set(begin, elements, offset, elements.length - offset);
	}

	public void set(double[] elements, int offset, int length) {
		set(0, elements, offset, length);
	}

	public void set(double[] elements, int offset) {
		set(elements, offset, elements.length - offset);
	}

	public void set(int begin, double[] elements) {
		set(begin, elements, 0, elements.length);
	}

	public void set(double[] elements) {
		set(0, elements);
	}

	@Override
	public DoubleArray clone() {
		return new DoubleArray(this);
	}

	@Override
	public int hashCode() {
		return buffer.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof DoubleArray)) {
			return false;
		}
		DoubleArray other = (DoubleArray) obj;
		return buffer.equals(other.buffer);
	}

	@Override
	protected Buffer buffer() {
		return buffer;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append('[');
		if (length > 0) {
			int index = 0;
			for (int limit = length - 1; index < limit; index++) {
				builder.append(buffer.get(index));
				builder.append(", ");
			}
			builder.append(buffer.get(index));
		}
		builder.append(']');
		return builder.toString();
	}
}
