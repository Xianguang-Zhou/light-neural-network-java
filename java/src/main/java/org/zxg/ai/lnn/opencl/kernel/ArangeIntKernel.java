/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl.kernel;

import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class ArangeIntKernel extends Kernel {

	public void execute(int start, int stop, int step, int repeat, IntArray result) {
		Calling c = call();
		c.arg(start).arg(stop).arg(step).arg(repeat).inOut(result);
		c.pass(new Range1D(result.length));
		c.execute();
	}
}
