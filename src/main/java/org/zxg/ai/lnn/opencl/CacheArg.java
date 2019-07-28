/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.llb.CLMemObjBinding;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class CacheArg extends Arg {

	public static enum Type {

		SHORT(2), INT(4), LONG(8), FLOAT(4), DOUBLE(8);

		public final int size;

		private Type(int size) {
			this.size = size;
		}
	};

	public final int size;
	private CLBuffer<?> clBuffer;

	public CacheArg(Type type, int number) {
		this(type.size * number);
	}

	public CacheArg(int size) {
		this.size = size;
	}

	@Override
	public void input(CLKernel kernel, CLCommandQueue queue) {
		clBuffer = kernel.getContext().createBuffer(size, CLMemObjBinding.CL_MEM_READ_WRITE);
		kernel.putArg(clBuffer);
	}

	@Override
	public void close() {
		clBuffer.release();
		clBuffer = null;
	}
}
