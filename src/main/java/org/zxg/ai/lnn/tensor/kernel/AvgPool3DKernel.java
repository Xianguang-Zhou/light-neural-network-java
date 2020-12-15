/*
 * Copyright (c) 2020, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
package org.zxg.ai.lnn.tensor.kernel;

import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.tensor.Tensor;
import org.zxg.ai.lnn.tuple.IntTuple3;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class AvgPool3DKernel extends Kernel {

	public void execute(Tensor input, IntTuple3 kernelSize, IntTuple3 stride, IntTuple3 padding,
			boolean countIncludePad, Integer divisorOverride, Tensor result) {
		FloatArray resultData = result.flatData();
		Calling c = call();
		c.arg(kernelSize.e0).arg(kernelSize.e1).arg(kernelSize.e2);
		c.arg(stride.e0).arg(stride.e1).arg(stride.e2);
		c.arg(padding.e0).arg(padding.e1).arg(padding.e2);
		c.arg((short) (countIncludePad ? 1 : 0));
		c.arg(divisorOverride != null ? divisorOverride.intValue() : 0);
		c.in(input.shape()).in(result.shape());
		c.in(input.dimSizes()).in(result.dimSizes());
		c.in(input.flatData()).out(resultData);
		c.pass(new Range1D(resultData.length));
		c.execute();
	}
}
