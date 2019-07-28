/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor.kernel;

import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class SliceAssignKernel extends Kernel {

	public void execute(IntArray begin, Tensor source, Tensor result) {
		FloatArray sourceData = source.flatData();
		Calling c = call();
		c.arg(result.ndim()).in(begin);
		c.in(source.dimSizes()).in(result.dimSizes());
		c.in(sourceData).inOut(result.flatData());
		c.pass(new Range1D(sourceData.length));
		c.execute();
	}
}
