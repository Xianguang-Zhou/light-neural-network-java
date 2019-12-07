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

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.llb.CLMemObjBinding;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class BufferArg extends Arg {

	public static enum Type {
		IN, OUT, IN_OUT;
	}

	public final Type type;
	private Buffer directBuffer;
	private CLBuffer<?> clBuffer;

	public BufferArg(Buffer directBuffer, Type type) {
		if (null == directBuffer || null == type) {
			throw new NullPointerException();
		}
		if (!directBuffer.isDirect()) {
			throw new LnnCLException();
		}
		this.directBuffer = directBuffer;
		this.type = type;
	}

	public boolean isIn() {
		return type == Type.IN || type == Type.IN_OUT;
	}

	public boolean isOut() {
		return type == Type.OUT || type == Type.IN_OUT;
	}

	protected static int argTypeToFlags(Type type) {
		switch (type) {
		case IN:
			return CLMemObjBinding.CL_MEM_READ_ONLY;
		case OUT:
			return CLMemObjBinding.CL_MEM_WRITE_ONLY;
		default:
			return CLMemObjBinding.CL_MEM_READ_WRITE;
		}
	}

	@Override
	public void input(CLKernel kernel, CLCommandQueue queue) {
		clBuffer = kernel.getContext().createBuffer(directBuffer, argTypeToFlags(type));
		kernel.putArg(clBuffer);
		if (isIn()) {
			queue.putWriteBuffer(clBuffer, false);
		}
	}

	@Override
	public void output(CLCommandQueue queue) {
		if (isOut()) {
			queue.putReadBuffer(clBuffer, false);
		}
	}

	@Override
	public void close() {
		clBuffer.release();
		clBuffer = null;
		directBuffer = null;
	}
}
