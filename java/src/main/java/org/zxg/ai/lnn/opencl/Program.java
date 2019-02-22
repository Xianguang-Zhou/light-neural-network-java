/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;

import org.zxg.ai.lnn.LnnIOException;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.CLProgram;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Program implements Closeable {

	private static CLContext DEFAULT_CONTEXT;
	private static CLDevice DEFAULT_DEVICE;

	public static CLContext defaultContext() {
		if (null == DEFAULT_CONTEXT) {
			DEFAULT_CONTEXT = CLContext.create();
		}
		return DEFAULT_CONTEXT;
	}

	public static CLDevice defaultDevice() {
		if (null == DEFAULT_DEVICE) {
			DEFAULT_DEVICE = defaultContext().getMaxFlopsDevice();
		}
		return DEFAULT_DEVICE;
	}

	private CLProgram program;

	protected Program() {
	}

	public Program(CLContext ctx) throws LnnIOException, LnnCLException {
		init(ctx);
	}

	public Program(CLContext ctx, String name) throws LnnIOException, LnnCLException {
		init(ctx, name);
	}

	public Program(CLContext ctx, InputStream in) throws LnnIOException, LnnCLException {
		init(ctx, in);
	}

	protected final void init(CLContext ctx) throws LnnIOException, LnnCLException {
		init(ctx, this.getClass().getSimpleName() + ".cl");
	}

	protected final void init(CLContext ctx, String name) throws LnnIOException, LnnCLException {
		init(ctx, this.getClass().getResourceAsStream(name));
	}

	protected final void init(CLContext ctx, InputStream in) throws LnnIOException, LnnCLException {
		try {
			this.program = ctx.createProgram(in);
			this.program.build();
		} catch (IOException e) {
			throw new LnnIOException(e);
		} catch (CLException e) {
			throw new LnnCLException(e);
		}
	}

	public Calling call() {
		return call("run");
	}

	public Calling call(String function) {
		return new Calling(program, function);
	}

	@Override
	public void close() {
		this.program.release();
	}
}
