/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.nio.LongBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class LongArray extends BufferArray {

	private final LongBuffer buffer;
	public final int length;

	public static void copy(LongArray src, int srcPos, LongArray dest, int destPos, int length) {
		for (int i = 0; i < length; i++) {
			dest.set(destPos + i, src.get(srcPos + i));
		}
	}

	public LongArray(int length) {
		this.buffer = Buffers.newDirectLongBuffer(length);
		this.length = length;
	}

	public LongArray(long[] elements) {
		this.buffer = Buffers.newDirectLongBuffer(elements);
		this.length = elements.length;
	}

	public LongArray(LongArray other) {
		this.buffer = Buffers.copyLongBuffer(other.buffer);
		this.length = other.length;
	}

	public LongArray(LongBuffer buffer) {
		this.buffer = Buffers.copyLongBuffer(buffer);
		this.length = this.buffer.capacity() / Buffers.SIZEOF_LONG;
	}

	public long get(int index) {
		return buffer.get(index);
	}

	public void set(int index, long value) {
		buffer.put(index, value);
	}

	public long[] get(int begin, int end) {
		long[] elements = new long[end - begin];
		for (int i = 0; i < elements.length; i++) {
			elements[i] = buffer.get(begin + i);
		}
		return elements;
	}

	public long[] get() {
		return get(0, length);
	}

	public void set(int begin, long[] elements, int offset, int length) {
		for (int i = 0; i < length; i++) {
			buffer.put(begin + i, elements[offset + i]);
		}
	}

	public void set(int begin, long[] elements, int offset) {
		set(begin, elements, offset, elements.length - offset);
	}

	public void set(long[] elements, int offset, int length) {
		set(0, elements, offset, length);
	}

	public void set(long[] elements, int offset) {
		set(elements, offset, elements.length - offset);
	}

	public void set(int begin, long[] elements) {
		set(begin, elements, 0, elements.length);
	}

	public void set(long[] elements) {
		set(0, elements);
	}

	@Override
	public LongArray clone() {
		return new LongArray(this);
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
		if (!(obj instanceof LongArray)) {
			return false;
		}
		LongArray other = (LongArray) obj;
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
