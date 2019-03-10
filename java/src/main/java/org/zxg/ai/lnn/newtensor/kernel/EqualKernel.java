/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.newtensor.kernel;

import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.opencl.ShortArray;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class EqualKernel extends Kernel {

	public boolean execute(float precision, FloatArray left, FloatArray right) {
		ShortArray result = new ShortArray(new short[] { 0 });
		Calling c = call();
		c.arg(precision).in(left).in(right).inOut(result);
		c.pass(new Range1D(left.length));
		c.execute();
		return 0 == result.get(0);
	}
}
