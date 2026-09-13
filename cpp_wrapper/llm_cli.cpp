#include <iostream>
#include <sstream>
#include <string>

extern "C" bool InitLLM(const char* model_path);
extern "C" int GenerateText(const char* prompt, char* output_buffer, int max_output_len, int max_new_tokens);
extern "C" void ShutdownLLM();

int main(int argc, char** argv) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string prompt = ss.str();

    if (prompt.empty()) { std::cerr << "usage: pipe prompt text via stdin" << std::endl; return 1; }

    if (!InitLLM("qwen2.5-0.5b-q4_k_m-v2.gguf")) {
        std::cerr << "InitLLM failed" << std::endl;
        return 1;
    }

    char buffer[4096];
    int len = GenerateText(prompt.c_str(), buffer, sizeof(buffer), 60);
    if (len > 0) {
        std::cout << buffer;
    }

    ShutdownLLM();
    return 0;
}