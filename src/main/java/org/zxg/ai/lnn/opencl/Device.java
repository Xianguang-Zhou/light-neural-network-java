/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
package org.zxg.ai.lnn.opencl;

import java.io.Closeable;
import java.util.HashMap;
import java.util.Map;

import com.jogamp.opencl.CLDevice;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Device implements Closeable {

	private static Device DEFAULT_DEVICE;

	public static Device defaultDevice() {
		if (null == DEFAULT_DEVICE) {
			synchronized (Device.class) {
				if (null == DEFAULT_DEVICE) {
					DEFAULT_DEVICE = new Device(Program.defaultDevice());
				}
			}
		}
		return DEFAULT_DEVICE;
	}

	public final CLDevice clDevice;
	private Map<Class<? extends Kernel>, Kernel> kernels;

	public Device(CLDevice clDevice) {
		this.clDevice = clDevice;
		kernels = new HashMap<>();
	}

	@SuppressWarnings("unchecked")
	public <T extends Kernel> T kernel(Class<T> type) {
		Kernel kernel = kernels.get(type);
		if (null == kernel) {
			synchronized (kernels) {
				kernel = kernels.get(type);
				if (null == kernel) {
					try {
						kernel = type.newInstance();
						kernel.init(clDevice);
						kernels.put(type, kernel);
					} catch (InstantiationException | IllegalAccessException e) {
						throw new LnnCLException(e);
					}
				}
			}
		}
		return (T) kernel;
	}

	@Override
	public void close() {
		for (Kernel k : kernels.values()) {
			k.close();
		}
	}
}
