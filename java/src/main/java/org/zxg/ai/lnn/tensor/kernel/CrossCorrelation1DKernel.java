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
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class CrossCorrelation1DKernel extends Kernel {

	public void execute(Tensor input, Tensor weight, int stride, int padding, int dilation, int groups, Tensor result) {
		FloatArray resultData = result.flatData();
		Calling c = call();
		c.arg(stride).arg(padding).arg(dilation).arg(groups);
		c.in(input.shape()).in(weight.shape()).in(result.shape());
		c.in(input.dimSizes()).in(weight.dimSizes()).in(result.dimSizes());
		c.in(input.flatData()).in(weight.flatData()).out(resultData);
		c.pass(new Range1D(resultData.length));
		c.execute();
	}
}
