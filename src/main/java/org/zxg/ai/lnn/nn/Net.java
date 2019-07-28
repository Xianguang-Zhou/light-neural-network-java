/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.nn;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import org.zxg.ai.lnn.LnnException;
import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Net implements Component {

	private static final Pattern dotPattern = Pattern.compile("\\.");

	protected Map<String, Component> children = new LinkedHashMap<>();

	@Override
	public void addComponent(String name, Component component) {
		if (name.isEmpty()) {
			throw new LnnException("Component name can not be empty string \"\".");
		}
		if (name.contains(".")) {
			throw new LnnException("Component name can not contain \".\".");
		}
		children.put(name, component);
	}

	@Override
	public void apply(Consumer<Component> function) {
		for (Component child : children.values()) {
			child.apply(function);
		}
		function.accept(this);
	}

	@Override
	public Component child(String name) {
		return children.get(name);
	}

	@Override
	public Iterable<Component> children() {
		return children.values();
	}

	@Override
	public Iterable<Component> components() {
		Map<String, Component> result = new LinkedHashMap<>();
		namedComponents(result);
		return result.values();
	}

	@Override
	public void loadState(Map<String, Variable> state, boolean strict) {
		if (strict && !state.keySet().equals(this.state().keySet())) {
			throw new LnnException(
					"The keys of state must exactly match the keys returned by this component’s state() function.");
		}
		for (Entry<String, Variable> entry : state.entrySet()) {
			registerParameter(entry.getKey(), entry.getValue());
		}
	}

	@Override
	public Iterable<Entry<String, Component>> namedChildren() {
		return children.entrySet();
	}

	@Override
	public Iterable<Entry<String, Component>> namedComponents(Map<String, Component> memo, String prefix) {
		if (!memo.containsValue(this)) {
			memo.put(prefix, this);
			if (!prefix.isEmpty()) {
				prefix += ".";
			}
			for (Entry<String, Component> childEntry : this.children.entrySet()) {
				String childPrefix = prefix + childEntry.getKey();
				childEntry.getValue().namedComponents(memo, childPrefix);
			}
		}
		return memo.entrySet();
	}

	@Override
	public Iterable<Entry<String, Variable>> namedParameters(String prefix, boolean recurse) {
		if (recurse) {
			return state(prefix).entrySet();
		} else {
			return Collections.emptySet();
		}
	}

	@Override
	public Variable parameter(String name) {
		if (name.isEmpty()) {
			throw new LnnException("Parameter name can not be empty string \"\".");
		}
		String[] nameFragments = dotPattern.split(name);
		if (nameFragments.length < 2) {
			return null;
		}
		Component component = this;
		int lastIndexOfNameFragments = nameFragments.length - 1;
		for (int i = 0; i < lastIndexOfNameFragments; i++) {
			component = component.child(nameFragments[i]);
			if (null == component) {
				return null;
			}
		}
		if (this != component) {
			return component.parameter(nameFragments[lastIndexOfNameFragments]);
		} else {
			return null;
		}
	}

	@Override
	public Iterable<Variable> parameters(boolean recurse) {
		if (recurse) {
			return state().values();
		} else {
			return Collections.emptySet();
		}
	}

	@Override
	public void registerParameter(String name, Variable parameter) {
		if (name.isEmpty()) {
			throw new LnnException("Parameter name can not be empty string \"\".");
		}
		String[] nameFragments = dotPattern.split(name);
		if (nameFragments.length < 2) {
			return;
		}
		Component component = this;
		int lastIndexOfNameFragments = nameFragments.length - 1;
		for (int i = 0; i < lastIndexOfNameFragments; i++) {
			component = component.child(nameFragments[i]);
			if (null == component) {
				return;
			}
		}
		if (this != component) {
			component.registerParameter(nameFragments[lastIndexOfNameFragments], parameter);
		}
	}

	@Override
	public Map<String, Variable> state(Map<String, Variable> destination, String prefix) {
		if (!prefix.isEmpty()) {
			prefix += ".";
		}
		for (Entry<String, Component> childEntry : this.children.entrySet()) {
			String childPrefix = prefix + childEntry.getKey();
			state(destination, childPrefix);
		}
		return destination;
	}

	@Override
	public void zeroGradient() {
		for (Component child : children.values()) {
			child.zeroGradient();
		}
	}

	private String toString(int indent) {
		StringBuilder builder = new StringBuilder();
		builder.append(getClass().getSimpleName());
		builder.append("(\n");
		for (Entry<String, Component> entry : children.entrySet()) {
			for (int i = 0; i < indent; i++) {
				builder.append(' ');
			}
			builder.append("  (");
			String childName = entry.getKey();
			builder.append(childName);
			builder.append("): ");
			Component child = entry.getValue();
			if (child instanceof Net) {
				builder.append(((Net) child).toString(indent + childName.length() + 6));
			} else {
				builder.append(child);
			}
			builder.append('\n');
		}
		for (int i = 0; i < indent; i++) {
			builder.append(' ');
		}
		builder.append(')');
		return builder.toString();
	}

	@Override
	public String toString() {
		return toString(0);
	}
}
