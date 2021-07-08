/*
 * Copyright (c) 2021, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
import org.zxg.ai.lnn.tensor.IntTensor;
import org.zxg.ai.lnn.tensor.Tensor;
import org.zxg.ai.lnn.tuple.IntTuple2;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class MaxPool2DKernel extends Kernel {

	public void execute(Tensor input, IntTuple2 kernelSize, IntTuple2 stride, IntTuple2 padding, IntTuple2 dilation,
			Tensor result, IntTensor indices) {
		FloatArray resultData = result.flatData();
		Calling c = call();
		c.in(input.flatData()).in(input.shape()).in(input.dimSizes());
		c.out(resultData).in(result.dimSizes());
		if (indices != null) {
			c.out(indices.flatData());
		} else {
			c.nullPtr();
		}
		c.arg(kernelSize.e0).arg(kernelSize.e1);
		c.arg(stride.e0).arg(stride.e1);
		c.arg(padding.e0).arg(padding.e1);
		c.arg(dilation.e0).arg(dilation.e1);
		c.pass(new Range1D(resultData.length));
		c.execute();
	}
}
