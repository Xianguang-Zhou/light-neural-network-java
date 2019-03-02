/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.nio.FloatBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class FloatArray extends BufferArray {

	private final FloatBuffer buffer;
	public final int length;

	public static void copy(FloatArray src, int srcPos, FloatArray dest, int destPos, int length) {
		for (int i = 0; i < length; i++) {
			dest.set(destPos + i, src.get(srcPos + i));
		}
	}

	public FloatArray(int length) {
		this.buffer = Buffers.newDirectFloatBuffer(length);
		this.length = length;
	}

	public FloatArray(float[] elements) {
		this.buffer = Buffers.newDirectFloatBuffer(elements);
		this.length = elements.length;
	}

	public FloatArray(FloatArray other) {
		this.buffer = Buffers.copyFloatBuffer(other.buffer);
		this.length = other.length;
	}

	public FloatArray(FloatBuffer buffer) {
		this.buffer = Buffers.copyFloatBuffer(buffer);
		this.length = this.buffer.capacity() / Buffers.SIZEOF_FLOAT;
	}

	public float get(int index) {
		return buffer.get(index);
	}

	public void set(int index, float value) {
		buffer.put(index, value);
	}

	public float[] get(int begin, int end) {
		float[] elements = new float[end - begin];
		for (int i = 0; i < elements.length; i++) {
			elements[i] = buffer.get(begin + i);
		}
		return elements;
	}

	public float[] get() {
		return get(0, length);
	}

	public void set(int begin, float[] elements, int offset, int length) {
		for (int i = 0; i < length; i++) {
			buffer.put(begin + i, elements[offset + i]);
		}
	}

	public void set(int begin, float[] elements, int offset) {
		set(begin, elements, offset, elements.length - offset);
	}

	public void set(float[] elements, int offset, int length) {
		set(0, elements, offset, length);
	}

	public void set(float[] elements, int offset) {
		set(elements, offset, elements.length - offset);
	}

	public void set(int begin, float[] elements) {
		set(begin, elements, 0, elements.length);
	}

	public void set(float[] elements) {
		set(0, elements);
	}

	@Override
	public FloatArray clone() {
		return new FloatArray(this);
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
		if (!(obj instanceof FloatArray)) {
			return false;
		}
		FloatArray other = (FloatArray) obj;
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
