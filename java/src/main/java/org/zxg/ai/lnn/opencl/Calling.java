/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.util.LinkedList;
import java.util.List;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.llb.CLMemObjBinding;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Calling {

	private CLProgram program;
	private String function;
	private List<Arg> args = new LinkedList<>();
	private List<Range> ranges = new LinkedList<>();
	private CLDevice device;

	public Calling(CLProgram program, String function) {
		this.program = program;
		this.function = function;
	}

	public Calling arg(Arg arg) {
		this.args.add(arg);
		return this;
	}

	public Calling in(Buffer directBuffer) {
		return arg(Arg.in(directBuffer));
	}

	public Calling out(Buffer directBuffer) {
		return arg(Arg.out(directBuffer));
	}

	public Calling inOut(Buffer directBuffer) {
		return arg(Arg.inOut(directBuffer));
	}

	public Calling pass(Range range) {
		this.ranges.add(range);
		return this;
	}

	public Calling at(CLDevice device) throws LnnCLException {
		if (!program.getContext().equals(device.getContext())) {
			throw new LnnCLException();
		}
		this.device = device;
		return this;
	}

	protected static int argTypeToFlags(Arg.Type type) {
		switch (type) {
		case IN:
			return CLMemObjBinding.CL_MEM_READ_ONLY;
		case OUT:
			return CLMemObjBinding.CL_MEM_WRITE_ONLY;
		default:
			return CLMemObjBinding.CL_MEM_READ_WRITE;
		}
	}

	public void execute() throws LnnCLException {
		if (ranges.isEmpty() || null == device) {
			throw new NullPointerException();
		}
		try {
			CLCommandQueue queue = device.createCommandQueue();
			try {
				CLKernel kernel;
				synchronized (program) {
					kernel = program.createCLKernel(function);
				}
				try {
					CLContext ctx = kernel.getContext();
					for (Arg arg : args) {
						arg.clBuffer = ctx.createBuffer(arg.directBuffer, argTypeToFlags(arg.type));
						kernel.putArg(arg.clBuffer);
					}
					for (Arg arg : args) {
						if (arg.isIn()) {
							queue.putWriteBuffer(arg.clBuffer, false);
						}
					}
					if (ranges.size() > 1) {
						int passIdArgIndex = args.size(), passId = 0;
						for (Range range : ranges) {
							if (0 == passId) {
								kernel.putArg(passId);
							} else {
								kernel.setArg(passIdArgIndex, passId);
							}
							range.putToQueue(queue, kernel);
							passId++;
						}
					} else {
						ranges.get(0).putToQueue(queue, kernel);
					}
					for (Arg arg : args) {
						if (arg.isOut()) {
							queue.putReadBuffer(arg.clBuffer, false);
						}
					}
					queue.finish();
					for (Arg arg : args) {
						arg.clBuffer.release();
						arg.clBuffer = null;
					}
				} finally {
					kernel.release();
				}
			} finally {
				queue.release();
			}
		} catch (CLException e) {
			throw new LnnCLException(e);
		}
	}
}
