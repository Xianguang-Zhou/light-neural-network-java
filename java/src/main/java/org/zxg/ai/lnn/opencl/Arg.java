/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;

import com.jogamp.opencl.CLBuffer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Arg {

	public static enum Type {
		IN, OUT, IN_OUT;
	}

	public final Buffer directBuffer;
	public final Type type;
	CLBuffer<?> clBuffer;

	public Arg(Buffer directBuffer, Type type) {
		if (null == directBuffer || null == type) {
			throw new NullPointerException();
		}
		if (!directBuffer.isDirect()) {
			throw new LnnCLException();
		}
		this.directBuffer = directBuffer;
		this.type = type;
	}

	public static Arg in(Buffer directBuffer) {
		return new Arg(directBuffer, Type.IN);
	}

	public static Arg out(Buffer directBuffer) {
		return new Arg(directBuffer, Type.OUT);
	}

	public static Arg inOut(Buffer directBuffer) {
		return new Arg(directBuffer, Type.IN_OUT);
	}

	public boolean isIn() {
		return type == Type.IN || type == Type.IN_OUT;
	}

	public boolean isOut() {
		return type == Type.OUT || type == Type.IN_OUT;
	}

	public CLBuffer<?> clBuffer() {
		return clBuffer;
	}
}
