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
public class Range2D extends Range {

	private int globalWidth;
	private int globalHeight;
	private int localWidth;
	private int localHeight;

	public Range2D(int globalWidth, int globalHeight) {
		this(globalWidth, globalHeight, 1, 1);
	}

	public Range2D(int globalWidth, int globalHeight, int localWidth, int localHeight) {
		this.globalWidth = globalWidth;
		this.globalHeight = globalHeight;
		this.localWidth = localWidth;
		this.localHeight = localHeight;
	}

	@Override
	public void putToQueue(CLCommandQueue queue, CLKernel kernel) {
		queue.put2DRangeKernel(kernel, 0, 0, globalWidth, globalHeight, localWidth, localHeight);
	}
}
