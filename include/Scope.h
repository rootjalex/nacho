#pragma once

#include <iostream>
#include <map>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// #include "Debug.h"
#include "Error.h"

/** \file
 * Defines the Scope class, which is used for keeping track of names in a scope
 * while traversing IR
 */

namespace nacho {

// TODO: bring other parts of Scope.h that we need

/** Helper class for saving/restoring variable values on the stack, to allow
 * for early-exit that preserves correctness */
template <typename T>
struct ScopedValue {
    T &var;
    T old_value;
    /** Preserve the old value, restored at dtor time */
    ScopedValue(T &var) : var(var), old_value(var) {}
    /** Preserve the old value, then set the var to a new value. */
    ScopedValue(T &var, T new_value) : var(var), old_value(var) {
        var = new_value;
    }
    ~ScopedValue() { var = old_value; }
    operator T() const { return old_value; }
    // allow move but not copy
    ScopedValue(const ScopedValue &that) = delete;
    ScopedValue(ScopedValue &&that) noexcept = default;
};

} // namespace nacho
