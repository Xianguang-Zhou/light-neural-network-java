/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.nio.IntBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class IntArray extends BufferArray {

	private final IntBuffer buffer;
	public final int length;

	public IntArray(int length) {
		this.buffer = Buffers.newDirectIntBuffer(length);
		this.length = length;
	}

	public IntArray(int[] elements) {
		this.buffer = Buffers.newDirectIntBuffer(elements);
		this.length = elements.length;
	}

	public IntArray(IntArray other) {
		this.buffer = Buffers.copyIntBuffer(other.buffer);
		this.length = other.length;
	}

	public IntArray(IntBuffer buffer) {
		this.buffer = Buffers.copyIntBuffer(buffer);
		this.length = this.buffer.capacity() / Buffers.SIZEOF_INT;
	}

	public int get(int index) {
		return buffer.get(index);
	}

	public void set(int index, int value) {
		buffer.put(index, value);
	}

	public int[] get(int begin, int end) {
		int[] elements = new int[end - begin];
		for (int i = 0; i < elements.length; i++) {
			elements[i] = buffer.get(begin + i);
		}
		return elements;
	}

	public int[] get() {
		return get(0, length);
	}

	public void set(int begin, int[] elements, int offset, int length) {
		for (int i = 0; i < length; i++) {
			buffer.put(begin + i, elements[offset + i]);
		}
	}

	public void set(int begin, int[] elements, int offset) {
		set(begin, elements, offset, elements.length - offset);
	}

	public void set(int[] elements, int offset, int length) {
		set(0, elements, offset, length);
	}

	public void set(int[] elements, int offset) {
		set(elements, offset, elements.length - offset);
	}

	public void set(int begin, int[] elements) {
		set(begin, elements, 0, elements.length);
	}

	public void set(int[] elements) {
		set(0, elements);
	}

	@Override
	protected Object clone() {
		return new IntArray(this);
	}

	@Override
	public int hashCode() {
		return buffer.hashCode();
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
