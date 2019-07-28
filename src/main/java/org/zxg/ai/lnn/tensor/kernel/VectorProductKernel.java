/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor.kernel;

import org.zxg.ai.lnn.opencl.CacheArg;
import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class VectorProductKernel extends Kernel {

	public void execute(Tensor left, Tensor right, Tensor result) {
		final int cacheLength = right.shape().get(0);
		Calling c = call();
		c.arg(cacheLength);
		c.in(left.flatData()).in(right.flatData());
		c.cache(CacheArg.Type.FLOAT, cacheLength);
		c.out(result.flatData());
		c.pass(new Range1D(cacheLength));
		c.pass(new Range1D(1));
		c.execute();
	}
}
