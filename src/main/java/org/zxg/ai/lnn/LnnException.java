/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class LnnException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public LnnException() {
	}

	public LnnException(String message) {
		super(message);
	}

	public LnnException(Throwable cause) {
		super(cause);
	}

	public LnnException(String message, Throwable cause) {
		super(message, cause);
	}

	public LnnException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
