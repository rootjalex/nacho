#include "backend/output.h"

#include "Error.h"

#include <filesystem>

namespace nacho {
namespace backend {

namespace {
std::string &mutable_output_directory() {
    static std::string dir = "generated";
    return dir;
}
} // namespace

const std::string &output_directory() {
    return mutable_output_directory();
}

void set_output_directory(std::string dir) {
    mutable_output_directory() = std::move(dir);
}

void reset_output_directory() {
    std::error_code ec;
    std::filesystem::remove_all(output_directory(), ec);
    internal_assert(!ec) << "Could not clear output directory '" << output_directory()
                         << "': " << ec.message();

    std::filesystem::create_directories(output_directory(), ec);
    internal_assert(!ec) << "Could not create output directory '" << output_directory()
                         << "': " << ec.message();
}

} // namespace backend
} // namespace nacho
