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

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLKernel;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Range3D extends Range {

	private int globalWidth;
	private int globalHeight;
	private int globalDepth;
	private int localWidth;
	private int localHeight;
	private int localDepth;

	public Range3D(int globalWidth, int globalHeight, int globalDepth) {
		this(globalWidth, globalHeight, globalDepth, 1, 1, 1);
	}

	public Range3D(int globalWidth, int globalHeight, int globalDepth, int localWidth, int localHeight,
			int localDepth) {
		this.globalWidth = globalWidth;
		this.globalHeight = globalHeight;
		this.globalDepth = globalDepth;
		this.localWidth = localWidth;
		this.localHeight = localHeight;
		this.localDepth = localDepth;
	}

	@Override
	public void putToQueue(CLCommandQueue queue, CLKernel kernel) {
		queue.put3DRangeKernel(kernel, 0, 0, 0, globalWidth, globalHeight, globalDepth, localWidth, localHeight,
				localDepth);
	}
}
