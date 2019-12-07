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

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.LongBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Buffers extends com.jogamp.common.nio.Buffers {

	public static LongBuffer copyLongBuffer(LongBuffer orig) {
		return copyLongBufferAsByteBuffer(orig).asLongBuffer();
	}

	public static ByteBuffer copyLongBufferAsByteBuffer(final LongBuffer orig) {
		final int op0 = orig.position();
		final ByteBuffer dest = newDirectByteBuffer(orig.remaining() * SIZEOF_LONG);
		dest.asLongBuffer().put(orig);
		dest.rewind();
		orig.position(op0);
		return dest;
	}

	public static DoubleBuffer copyDoubleBuffer(DoubleBuffer orig) {
		return copyDoubleBufferAsByteBuffer(orig).asDoubleBuffer();
	}

	public static ByteBuffer copyDoubleBufferAsByteBuffer(final DoubleBuffer orig) {
		final int op0 = orig.position();
		final ByteBuffer dest = newDirectByteBuffer(orig.remaining() * SIZEOF_DOUBLE);
		dest.asDoubleBuffer().put(orig);
		dest.rewind();
		orig.position(op0);
		return dest;
	}
}
