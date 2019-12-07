/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
		final LongBuffer srcBuffer = src.buffer;
		final LongBuffer destBuffer = dest.buffer;
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

	public static LongArray copyOfRange(LongArray original, int from, int to) {
		int newLength = to - from;
		if (newLength < 0) {
			throw new IllegalArgumentException(from + " > " + to);
		}
		LongArray newArray = new LongArray(newLength);
		copy(original, from, newArray, 0, Math.min(original.length - from, newLength));
		return newArray;
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

	public void get(int begin, long[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.get(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
		}
	}

	public long[] get(int begin, int end) {
		long[] elements = new long[end - begin];
		get(begin, elements, 0, elements.length);
		return elements;
	}

	public long[] get() {
		return get(0, length);
	}

	public void set(int begin, long[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.put(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
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
