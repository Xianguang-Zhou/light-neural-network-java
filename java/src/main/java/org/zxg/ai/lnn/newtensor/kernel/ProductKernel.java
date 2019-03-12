/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.newtensor.kernel;

import org.zxg.ai.lnn.newtensor.Tensor;
import org.zxg.ai.lnn.opencl.CacheArg;
import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.opencl.Range2D;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class ProductKernel extends Kernel {

	public void execute(Tensor left, Tensor right, Tensor result) {
		FloatArray resultData = result.flatData();
		final int cacheWidth = resultData.length;
		final int cacheHeight = right.shape().get(0);
		Calling c = call();
		c.arg(left.ndim() - 1).arg(right.ndim()).arg(cacheHeight);
		c.in(left.dimSizes()).in(right.dimSizes()).in(result.dimSizes());
		c.in(left.flatData()).in(right.flatData());
		c.cache(CacheArg.Type.FLOAT, cacheWidth * cacheHeight);
		c.out(resultData);
		c.pass(new Range2D(cacheWidth, cacheHeight));
		c.pass(new Range1D(cacheWidth));
		c.execute();
	}
}
