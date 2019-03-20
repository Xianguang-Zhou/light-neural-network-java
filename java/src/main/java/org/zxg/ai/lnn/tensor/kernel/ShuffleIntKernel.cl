/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
typedef struct {
    long MASK;
    long MULTIPLIER;
    long ADDEND;
    long seed;
} Self;

float genNextFloat(Self* self, int index) {
    self->seed = self->seed * (index + 1);
    self->seed = (self->seed ^ self->MULTIPLIER) & self->MASK;
    self->seed = (self->seed * self->MULTIPLIER + self->ADDEND) & self->MASK;
    const int next24 = (int) (((unsigned long) self->seed) >> 24);
    return next24 / ((float) (1 << 24));
}

int genNextInt(Self* self, int index, int bound) {
    return (int) floor(genNextFloat(self, index) * bound);
}

__kernel void run(
    const long MASK,
    const long MULTIPLIER,
    const long ADDEND,
    const long seed,
    const int resultLength,
    __global int* cache,
    __global int* result,
    const int passId) {
    if (0 == passId) {
        Self self = {
            .MASK = MASK,
            .MULTIPLIER = MULTIPLIER,
            .ADDEND = ADDEND,
            .seed = seed
        };
        const size_t gid = get_global_id(0);
        cache[gid] = genNextInt(&self, gid, gid + 1);
    } else {
        for (size_t index = resultLength - 1; index > 0; index--) {
            const int swapIndex = cache[index];
            const int element = result[index];
            result[index] = result[swapIndex];
            result[swapIndex] = element;
        }
    }
}
