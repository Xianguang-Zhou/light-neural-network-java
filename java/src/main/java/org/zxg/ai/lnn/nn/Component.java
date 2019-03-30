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
import java.util.function.Consumer;

import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public interface Component {

	void addComponent(String name, Component component);

	void apply(Consumer<Component> function);

	Component child(String name);

	Iterable<Component> children();

	Iterable<Component> components();

	Variable[] forward(Variable... input);

	default void loadState(Map<String, Variable> state) {
		loadState(state, true);
	}

	void loadState(Map<String, Variable> state, boolean strict);

	Iterable<Entry<String, Component>> namedChildren();

	default Iterable<Entry<String, Component>> namedComponents() {
		return namedComponents(new LinkedHashMap<>());
	}

	default Iterable<Entry<String, Component>> namedComponents(Map<String, Component> memo) {
		return namedComponents(memo, "");
	}

	Iterable<Entry<String, Component>> namedComponents(Map<String, Component> memo, String prefix);

	default Iterable<Entry<String, Component>> namedComponents(String prefix) {
		return namedComponents(new LinkedHashMap<>(), prefix);
	}

	default Iterable<Map.Entry<String, Variable>> namedParameters() {
		return namedParameters("");
	}

	default Iterable<Map.Entry<String, Variable>> namedParameters(boolean recurse) {
		return namedParameters("", recurse);
	}

	default Iterable<Map.Entry<String, Variable>> namedParameters(String prefix) {
		return namedParameters(prefix, true);
	}

	Iterable<Map.Entry<String, Variable>> namedParameters(String prefix, boolean recurse);

	Variable parameter(String name);

	default Iterable<Variable> parameters() {
		return parameters(true);
	}

	Iterable<Variable> parameters(boolean recurse);

	void registerParameter(String name, Variable parameter);

	default Map<String, Variable> state() {
		return state(new LinkedHashMap<>());
	}

	default Map<String, Variable> state(Map<String, Variable> destination) {
		return state(destination, "");
	}

	Map<String, Variable> state(Map<String, Variable> destination, String prefix);

	default Map<String, Variable> state(String prefix) {
		return state(new LinkedHashMap<>(), prefix);
	}

	boolean training();

	void training(boolean mode);

	void zeroGradient();
}
