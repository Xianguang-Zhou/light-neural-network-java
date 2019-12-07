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
package org.zxg.ai.lnn.nn;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;

import org.zxg.ai.lnn.autograd.Constant;
import org.zxg.ai.lnn.autograd.Variable;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public interface Component {

	void addComponent(String name, Component component);

	void apply(Consumer<Component> function);

	Component child(String name);

	Iterable<Component> children();

	Iterable<Component> components();

	default Tensor[] forward(Tensor... input) {
		Variable[] variables = new Variable[input.length];
		for (int i = 0; i < input.length; i++) {
			variables[i] = new Constant(input[i]);
		}
		variables = forward(variables);
		Tensor[] output = new Tensor[variables.length];
		for (int i = 0; i < output.length; i++) {
			output[i] = variables[i].value();
		}
		return output;
	}

	default Variable[] forward(Variable... input) {
		return input;
	}

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

	void zeroGradient();
}
