/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.tensor;

import org.zxg.ai.lnn.LnnException;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class ShapeException extends LnnException {

	private static final long serialVersionUID = 1L;

	public ShapeException() {
	}

	public ShapeException(String message) {
		super(message);
	}

	public ShapeException(Throwable cause) {
		super(cause);
	}

	public ShapeException(String message, Throwable cause) {
		super(message, cause);
	}

	public ShapeException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
