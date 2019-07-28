/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
