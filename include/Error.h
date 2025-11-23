#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nacho {

class Error final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class ErrorReport {
  public:
    ErrorReport(const char *cond_str, const char *file, size_t line) {
        // Print the file path proceeding the root directory (inclusive).
        constexpr std::string_view ROOT_DIRECTORY = "nacho";
        std::string f(file);

        // TODO(cgyurgyik): Fix this hack.
        // Finds the last occurrence of `nacho` to conform with Github Actions,
        // where both the WORKSPACE and the REPOSITORY are named `nacho`.
        if (size_t pos = f.rfind(ROOT_DIRECTORY); pos != std::string::npos) {
            f = f.substr(pos + ROOT_DIRECTORY.length() + 1);
        }

        stream << "[internal] Error: ";
        stream << f << ":" << line << "\n";
        if (cond_str) {
            stream << "\n--> " << cond_str << "\n";
        }
    }

    ErrorReport &ref() { return *this; }

    template <typename T>
    ErrorReport &operator<<(const T &value) {
        stream << value;
        return *this;
    }

    [[noreturn]]
    ~ErrorReport() noexcept(false) {
        stream << "\n";
        // TODO: debug mode should do this.
        // std::cerr << stream.str();
        // abort();
        throw Error(stream.str());
    }

  private:
    std::ostringstream stream;
};

namespace detail {
// this is a syntax hack that enables placing a << operator after the .ref()
// method of the ErrorReport class. It can be any binary operator with lower
// precedence than << and higher than ?: (ternary). This also changes the
// semantics of `internal_assert(cond) << foo;` so that `foo` is only evaluated
// when the condition is false.
struct Voidifier {
    template <typename T>
    void operator&(T &) const {}
};
} // namespace detail

#define internal_assert(cond)                                                  \
    (cond) ? (void)0                                                           \
           : nacho::detail::Voidifier() &                                     \
                 nacho::ErrorReport(#cond, __FILE__, __LINE__).ref()

#define internal_error nacho::ErrorReport(nullptr, __FILE__, __LINE__)

} // namespace nacho
