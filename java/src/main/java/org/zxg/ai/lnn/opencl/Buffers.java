/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
