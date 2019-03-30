/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.nn;

import java.util.Collections;
import java.util.Map;
import java.util.Map.Entry;

import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public abstract class StatelessLayer extends Layer {

	@Override
	public void loadState(Map<String, Variable> state, boolean strict) {
	}

	@Override
	public Iterable<Entry<String, Variable>> namedParameters(String prefix, boolean recurse) {
		return Collections.emptySet();
	}

	@Override
	public Variable parameter(String name) {
		return null;
	}

	@Override
	public Iterable<Variable> parameters(boolean recurse) {
		return Collections.emptySet();
	}

	@Override
	public void registerParameter(String name, Variable parameter) {
	}

	@Override
	public Map<String, Variable> state(Map<String, Variable> destination, String prefix) {
		return destination;
	}

	@Override
	public boolean training() {
		return false;
	}

	@Override
	public void training(boolean mode) {
	}

	@Override
	public void zeroGradient() {
	}
}
