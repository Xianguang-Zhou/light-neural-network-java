/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.nn;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.zxg.ai.lnn.LnnException;
import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class StatefulLayer extends Layer {

	protected Map<String, Variable> state = new LinkedHashMap<>();

	@Override
	public void loadState(Map<String, Variable> state, boolean strict) {
		if (strict && !state.keySet().equals(this.state.keySet())) {
			throw new LnnException(
					"The keys of state must exactly match the keys returned by this component’s state() function.");
		}
		for (Entry<String, Variable> entry : state.entrySet()) {
			registerParameter(entry.getKey(), entry.getValue());
		}
	}

	@Override
	public Iterable<Entry<String, Variable>> namedParameters(String prefix, boolean recurse) {
		return state(prefix).entrySet();
	}

	@Override
	public Variable parameter(String name) {
		return state.get(name);
	}

	@Override
	public Iterable<Variable> parameters(boolean recurse) {
		return state.values();
	}

	@Override
	public void registerParameter(String name, Variable parameter) {
		if (name.isEmpty()) {
			throw new LnnException("Parameter name can not be empty string \"\".");
		}
		if (name.contains(".")) {
			throw new LnnException("Parameter name can not contain \".\".");
		}
		if (null == parameter.gradient()) {
			throw new LnnException("Parameter has no gradient.");
		}
		state.put(name, parameter);
	}

	@Override
	public Map<String, Variable> state(Map<String, Variable> destination, String prefix) {
		if (!prefix.isEmpty()) {
			prefix += ".";
		}
		for (Entry<String, Variable> entry : this.state.entrySet()) {
			destination.put(prefix + entry.getKey(), entry.getValue());
		}
		return destination;
	}

	@Override
	public void zeroGradient() {
		for (Variable parameter : state.values()) {
			parameter.zeroGradient();
		}
	}

	protected void appendExtraRepresentation(StringBuilder builder) {
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append(getClass().getSimpleName());
		builder.append('(');
		appendExtraRepresentation(builder);
		builder.append(')');
		return builder.toString();
	}
}
