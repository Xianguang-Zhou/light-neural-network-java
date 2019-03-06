/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.nio.ShortBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class ShortArray extends BufferArray {

	private final ShortBuffer buffer;
	public final int length;

	public static void copy(ShortArray src, int srcPos, ShortArray dest, int destPos, int length) {
		final ShortBuffer srcBuffer = src.buffer;
		final ShortBuffer destBuffer = dest.buffer;
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

	public ShortArray(int length) {
		this.buffer = Buffers.newDirectShortBuffer(length);
		this.length = length;
	}

	public ShortArray(short[] elements) {
		this.buffer = Buffers.newDirectShortBuffer(elements);
		this.length = elements.length;
	}

	public ShortArray(ShortArray other) {
		this.buffer = Buffers.copyShortBuffer(other.buffer);
		this.length = other.length;
	}

	public ShortArray(ShortBuffer buffer) {
		this.buffer = Buffers.copyShortBuffer(buffer);
		this.length = this.buffer.capacity() / Buffers.SIZEOF_SHORT;
	}

	public short get(int index) {
		return buffer.get(index);
	}

	public void set(int index, short value) {
		buffer.put(index, value);
	}

	public void get(int begin, short[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.get(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
		}
	}

	public short[] get(int begin, int end) {
		short[] elements = new short[end - begin];
		get(begin, elements, 0, elements.length);
		return elements;
	}

	public short[] get() {
		return get(0, length);
	}

	public void set(int begin, short[] elements, int offset, int length) {
		final int originalPosition = buffer.position();
		try {
			buffer.position(begin);
			buffer.put(elements, offset, length);
		} finally {
			buffer.position(originalPosition);
		}
	}

	public void set(int begin, short[] elements, int offset) {
		set(begin, elements, offset, elements.length - offset);
	}

	public void set(short[] elements, int offset, int length) {
		set(0, elements, offset, length);
	}

	public void set(short[] elements, int offset) {
		set(elements, offset, elements.length - offset);
	}

	public void set(int begin, short[] elements) {
		set(begin, elements, 0, elements.length);
	}

	public void set(short[] elements) {
		set(0, elements);
	}

	@Override
	public ShortArray clone() {
		return new ShortArray(this);
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
		if (!(obj instanceof ShortArray)) {
			return false;
		}
		ShortArray other = (ShortArray) obj;
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
