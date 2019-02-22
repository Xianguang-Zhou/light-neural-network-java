/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
