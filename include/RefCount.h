#pragma once

/** \file
 *
 * Support classes for reference-counting via intrusive shared
 * pointers.
 */

#include <atomic>
#include <cstdlib>

namespace nacho {

/** A class representing a reference count to be used with IntrusivePtr */
class RefCount {
    std::atomic<int32_t> count;

  public:
    RefCount() noexcept : count(0) {}
    int32_t increment() { return ++count; }
    // Increment and return new value
    int32_t decrement() { return --count; }
    // Decrement and return new value
    bool is_const_zero() const { return count == 0; }
};

/**
 * Because in this header we don't yet know how client classes store
 * their RefCount (and we don't want to depend on the declarations of
 * the client classes), any class that you want to hold onto via one
 * of these must provide implementations of ref_count and destroy,
 * which we forward-declare here.
 *
 * E.g. if you want to use IntrusivePtr<MyClass>, then you should
 * define something like this in MyClass.cpp (assuming MyClass has
 * a field: mutable RefCount ref_count):
 *
 * template<> RefCount &ref_count<MyClass>(const MyClass *c) noexcept {return
 * c->ref_count;} template<> void destroy<MyClass>(const MyClass *c) {delete c;}
 */
// @{
template <typename T>
RefCount &ref_count(const T *t) noexcept;
template <typename T>
void destroy(const T *t);
// @}

} // namespace nacho
