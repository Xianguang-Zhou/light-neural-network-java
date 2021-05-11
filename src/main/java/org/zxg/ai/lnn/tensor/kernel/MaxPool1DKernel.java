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

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class MaxPool1DKernel extends Kernel {

	public void execute(Tensor input, int kernelSize, int stride, int padding, int dilation, Tensor result,
			IntTensor indices) {
		FloatArray resultData = result.flatData();
		Calling c = call();
		c.arg(kernelSize).arg(stride).arg(padding).arg(dilation);
		c.in(input.shape()).in(result.shape());
		c.in(input.dimSizes()).in(result.dimSizes());
		c.in(input.flatData()).out(resultData);
		if (indices != null) {
			c.out(indices.flatData());
		} else {
			c.nullPtr();
		}
		c.pass(new Range1D(resultData.length));
		c.execute();
	}
}
