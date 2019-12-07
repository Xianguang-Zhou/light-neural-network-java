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
