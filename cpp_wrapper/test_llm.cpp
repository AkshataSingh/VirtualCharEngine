
#include <iostream>

extern "C" bool InitLLM(const char* model_path);
extern "C" int GenerateText(const char* prompt, char* output_buffer, int max_output_len, int max_new_tokens);
extern "C" void ShutdownLLM();

int main() {
    bool ok = InitLLM("qwen2.5-0.5b-q4_k_m-v2.gguf");
    std::cout << "Init: " << ok << std::endl;
    if (!ok) return 1;

    char buffer[2048];
    int len = GenerateText("You are a friendly game character. Greet the player.", buffer, sizeof(buffer), 60);
    std::cout << "Generated (" << len << " bytes): " << buffer << std::endl;

    ShutdownLLM();
    return 0;
}
