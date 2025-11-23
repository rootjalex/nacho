#pragma once

/** \file
 *
 * Support classes for reference-counting via intrusive shared
 * pointers.
 */

#include <utility>

#include "RefCount.h"

namespace nacho {

/** Intrusive shared pointers have a reference count (a
 * RefCount object) stored in the class itself. This is perhaps more
 * efficient than storing it externally, but more importantly, it
 * means it's possible to recover a reference-counted handle from the
 * raw pointer, and it's impossible to have two different reference
 * counts attached to the same raw object. Seeing as we pass around
 * raw pointers to concrete IRNodes and Expr's interchangeably, this
 * is a useful property.
 */
template <typename T>
struct IntrusivePtr {
  private:
    void incref(T *p) {
        if (p) {
            ref_count(p).increment();
        }
    }

    void decref(T *p) {
        if (p) {
            // Note that if the refcount is already zero, then we're
            // in a recursive destructor due to a self-reference (a
            // cycle), where the ref_count has been adjusted to remove
            // the counts due to the cycle. The next line then makes
            // the ref_count negative, which prevents actually
            // entering the destructor recursively.
            if (ref_count(p).decrement() == 0) {
                destroy(p);
            }
        }
    }

  protected:
    T *ptr = nullptr;

  public:
    /** Access the raw pointer in a variety of ways.
     * Note that a "const IntrusivePtr<T>" is not the same thing as an
     * IntrusivePtr<const T>. So the methods that return the ptr are
     * const, despite not adding an extra const to T. */
    // @{
    T *get() const { return ptr; }

    T &operator*() const { return *ptr; }

    T *operator->() const { return ptr; }
    // @}

    ~IntrusivePtr() { decref(ptr); }

    IntrusivePtr() = default;

    IntrusivePtr(T *p) : ptr(p) { incref(ptr); }

    IntrusivePtr(const IntrusivePtr<T> &other) noexcept : ptr(other.ptr) {
        incref(ptr);
    }

    IntrusivePtr(IntrusivePtr<T> &&other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    IntrusivePtr<T> &operator=(const IntrusivePtr<T> &other) {
        // Same-ptr but different-this happens frequently enough
        // to check for (see https://github.com/halide/Halide/pull/5412)
        if (other.ptr == ptr) {
            return *this;
        }
        // Other can be inside of something owned by this, so we
        // should be careful to incref other before we decref
        // ourselves.
        T *temp = other.ptr;
        incref(temp);
        decref(ptr);
        ptr = temp;
        return *this;
    }

    IntrusivePtr<T> &operator=(IntrusivePtr<T> &&other) noexcept {
        std::swap(ptr, other.ptr);
        return *this;
    }

    /* Handles can be null. This checks that. */
    bool defined() const { return ptr != nullptr; }

    /* Check if two handles point to the same ptr. This is
     * equality of reference, not equality of value. */
    bool same_as(const IntrusivePtr &other) const { return ptr == other.ptr; }

    bool operator<(const IntrusivePtr<T> &other) const {
        return ptr < other.ptr;
    }
};

} // namespace nacho
