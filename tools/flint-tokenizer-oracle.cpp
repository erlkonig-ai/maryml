// Vocabulary-only oracle against llama.cpp b11853 / 1ddffe9f70a63e2d260d4bfad9ad200009007f62.
// Compile with the matching public headers and retained libllama; no common_init,
// llama_backend_init, ggml_backend_load*, llama_init_from_model, or tensor callback.
#include "llama.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static std::string hex(const std::string & bytes) {
    const char * digits = "0123456789abcdef";
    std::string out;
    for (unsigned char b : bytes) { out += digits[b >> 4]; out += digits[b & 15]; }
    return out;
}
static std::string unhex(const std::string & text) {
    if (text == "-") return {};
    require(text.size() <= 131072 && text.size() % 2 == 0, "invalid hex input length");
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::runtime_error("invalid hex input");
    };
    std::string out;
    for (size_t i = 0; i < text.size(); i += 2) out += char((nibble(text[i]) << 4) | nibble(text[i + 1]));
    return out;
}
static std::string piece(const llama_vocab * vocab, llama_token id, bool controls) {
    char small[256];
    int32_t n = llama_token_to_piece(vocab, id, small, sizeof(small), 0, controls);
    if (n >= 0) return {small, size_t(n)};
    require(n >= -65536, "native piece exceeds 64 KiB");
    std::string result(size_t(-n), '\0');
    require(llama_token_to_piece(vocab, id, result.data(), result.size(), 0, controls) == -n, "unstable native piece size");
    return result;
}
static void emit(const llama_vocab * vocab, size_t index, const std::vector<llama_token> & ids) {
    std::cout << "{\"case\":" << index << ",\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
    std::cout << ']';
    for (bool controls : {false, true}) {
        std::string whole;
        std::cout << (controls ? ",\"pieces_all_hex\":[" : ",\"pieces_visible_hex\":[");
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto bytes = piece(vocab, ids[i], controls);
            whole += bytes;
            require(whole.size() <= 1024 * 1024, "decoded sequence too large");
            if (i) std::cout << ',';
            std::cout << '"' << hex(bytes) << '"';
        }
        std::cout << ']' << (controls ? ",\"all_hex\":\"" : ",\"visible_hex\":\"") << hex(whole) << '"';
    }
    std::cout << ",\"native_eog\":[";
    for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << (llama_vocab_is_eog(vocab, ids[i]) ? "true" : "false"); }
    std::cout << "]}\n";
}

int main(int argc, char ** argv) try {
    require(argc == 2, "usage: flint-tokenizer-oracle METADATA-ONLY.gguf < requests.txt");
    require(ggml_backend_reg_count() == 0, "unexpected registered backend before oracle");
    const int fd = open(argv[1], O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    require(fd >= 0, "cannot open metadata snapshot");
    struct stat before{}, after{};
    require(fstat(fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_size == 5437024,
        "expected exactly the 5,437,024-byte zero-tensor metadata snapshot");
    std::vector<char> data(size_t(before.st_size));
    size_t done = 0;
    while (done < data.size()) {
        const ssize_t n = read(fd, data.data() + done, data.size() - done);
        require(n > 0, "truncated metadata snapshot"); done += size_t(n);
    }
    require(fstat(fd, &after) == 0 && before.st_size == after.st_size
        && before.st_mtim.tv_sec == after.st_mtim.tv_sec && before.st_mtim.tv_nsec == after.st_mtim.tv_nsec,
        "metadata snapshot changed while reading");
    close(fd);
    require(std::string(data.data(), 4) == "GGUF" && data[4] == 3, "wrong GGUF header");
    for (size_t i = 8; i < 16; ++i) require(data[i] == 0, "snapshot must contain zero tensors");
    gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_buffer(data.data(), data.size(), gp), gguf_free);
    require(bool(metadata) && gguf_get_n_tensors(metadata.get()) == 0 && gguf_get_n_kv(metadata.get()) == 43, "wrong metadata snapshot structure");
    ggml_backend_dev_t no_devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.vocab_only = true; params.devices = no_devices; params.n_gpu_layers = 0;
    params.split_mode = LLAMA_SPLIT_MODE_NONE; params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false; params.check_tensors = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_init_from_user(metadata.get(), nullptr, nullptr, params), llama_model_free);
    require(bool(model), "native vocab-only initialization failed");
    require(ggml_backend_reg_count() == 0, "oracle unexpectedly registered a backend");
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab) == 150144, "native vocabulary size differs");
    size_t count = 0;
    while (std::cin.peek() != EOF) {
        std::string line;
        char c;
        while (std::cin.get(c) && c != '\n') { require(line.size() < 262144, "oversized request line"); line += c; }
        require(++count <= 256, "too many requests");
        std::istringstream input(line);
        char kind; size_t index; input >> kind >> index;
        require(bool(input), "invalid request header");
        std::vector<llama_token> ids;
        if (kind == 'E') {
            int add, parse; std::string encoded; input >> add >> parse >> encoded;
            require(bool(input) && (add == 0 || add == 1) && (parse == 0 || parse == 1), "invalid encode flags");
            const auto text = unhex(encoded);
            const int32_t needed = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add, parse);
            require(needed <= 0 && needed >= -65536, "native token count exceeds limit");
            ids.resize(size_t(-needed));
            const int32_t n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), add, parse);
            require(n == -needed, "unstable native token count");
        } else if (kind == 'D') {
            size_t n; input >> n; require(bool(input) && n <= 65536, "invalid decode count");
            ids.resize(n);
            for (auto & id : ids) { input >> id; require(bool(input) && id >= 0 && id < 150144, "invalid decode token"); }
        } else { throw std::runtime_error("unknown request kind"); }
        std::string extra; require(!(input >> extra), "trailing request data");
        emit(vocab, index, ids);
    }
    require(ggml_backend_reg_count() == 0, "oracle unexpectedly registered a backend");
    return 0;
} catch (const std::exception & error) { std::cerr << "oracle: " << error.what() << '\n'; return 1; }
