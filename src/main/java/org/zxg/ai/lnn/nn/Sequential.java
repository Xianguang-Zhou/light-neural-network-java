/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.nn;

import java.util.AbstractList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;

import org.zxg.ai.lnn.autograd.Variable;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Sequential extends Net implements List<Component> {

	private class ComponentList extends AbstractList<Component> {

		@Override
		public Component get(int index) {
			for (Component component : children.values()) {
				if (0 == index--) {
					return component;
				}
			}
			throw new IndexOutOfBoundsException();
		}

		@Override
		public Component remove(int index) {
			for (String name : children.keySet()) {
				if (0 == index--) {
					return children.remove(name);
				}
			}
			throw new IndexOutOfBoundsException();
		}

		@Override
		public Component set(int index, Component element) {
			for (String name : children.keySet()) {
				if (0 == index--) {
					return children.replace(name, element);
				}
			}
			throw new IndexOutOfBoundsException();
		}

		@Override
		public int size() {
			return children.size();
		}
	}

	private ComponentList componentList = new ComponentList();

	public Sequential(Component... children) {
		for (int i = 0; i < children.length; i++) {
			addComponent(Integer.toString(i), children[i]);
		}
	}

	public Sequential(Map<String, Component> children) {
		for (Entry<String, Component> entry : children.entrySet()) {
			addComponent(entry.getKey(), entry.getValue());
		}
	}

	@Override
	public boolean add(Component e) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void add(int index, Component element) {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean addAll(Collection<? extends Component> c) {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean addAll(int index, Collection<? extends Component> c) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void clear() {
		children.clear();
	}

	@Override
	public boolean contains(Object o) {
		return children.containsValue(o);
	}

	@Override
	public boolean containsAll(Collection<?> c) {
		return componentList.containsAll(c);
	}

	@Override
	public Variable[] forward(Variable... input) {
		for (Component child : children()) {
			input = child.forward(input);
		}
		return input;
	}

	@Override
	public Tensor[] forward(Tensor... input) {
		for (Component child : children()) {
			input = child.forward(input);
		}
		return input;
	}

	@Override
	public Component get(int index) {
		return componentList.get(index);
	}

	@Override
	public int indexOf(Object o) {
		return componentList.indexOf(o);
	}

	@Override
	public boolean isEmpty() {
		return children.isEmpty();
	}

	@Override
	public Iterator<Component> iterator() {
		return componentList.iterator();
	}

	@Override
	public int lastIndexOf(Object o) {
		return componentList.lastIndexOf(o);
	}

	@Override
	public ListIterator<Component> listIterator() {
		return componentList.listIterator();
	}

	@Override
	public ListIterator<Component> listIterator(int index) {
		return componentList.listIterator(index);
	}

	@Override
	public Component remove(int index) {
		return componentList.remove(index);
	}

	@Override
	public boolean remove(Object o) {
		return componentList.remove(o);
	}

	@Override
	public boolean removeAll(Collection<?> c) {
		return componentList.removeAll(c);
	}

	@Override
	public boolean retainAll(Collection<?> c) {
		return componentList.retainAll(c);
	}

	@Override
	public Component set(int index, Component element) {
		return componentList.set(index, element);
	}

	@Override
	public int size() {
		return componentList.size();
	}

	@Override
	public List<Component> subList(int fromIndex, int toIndex) {
		return componentList.subList(fromIndex, toIndex);
	}

	@Override
	public Object[] toArray() {
		return componentList.toArray();
	}

	@Override
	public <T> T[] toArray(T[] a) {
		return componentList.toArray(a);
	}
}
