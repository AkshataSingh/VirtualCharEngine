#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "llama.h"

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;

extern "C" {

__declspec(dllexport) bool InitLLM(const char* model_path) {
    try {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        g_model = llama_model_load_from_file(model_path, model_params);
        if (!g_model) {
            std::ofstream log("C:\\VirtualCharEngine\\llm_error.log");
            log << "llama_model_load_from_file returned null for path: " << model_path << std::endl;
            return false;
        }

        g_vocab = llama_model_get_vocab(g_model);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        g_ctx = llama_init_from_model(g_model, ctx_params);
        if (!g_ctx) {
            std::ofstream log("C:\\VirtualCharEngine\\llm_error.log");
            log << "llama_init_from_model returned null" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::ofstream log("C:\\VirtualCharEngine\\llm_error.log");
        log << "InitLLM exception: " << e.what() << std::endl;
        return false;
    }
}

__declspec(dllexport) int GenerateText(const char* prompt, char* output_buffer, int max_output_len, int max_new_tokens) {
    if (!g_ctx || !g_model) return -1;
    try {
        int n_prompt = -llama_tokenize(g_vocab, prompt, (int)strlen(prompt), nullptr, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt);
        llama_tokenize(g_vocab, prompt, (int)strlen(prompt), prompt_tokens.data(), n_prompt, true, true);

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

        std::string result;
        int n_generated = 0;

        while (n_generated < max_new_tokens) {
            if (llama_decode(g_ctx, batch) != 0) break;

            int n_vocab = llama_vocab_n_tokens(g_vocab);
            float* logits = llama_get_logits_ith(g_ctx, batch.n_tokens - 1);

            llama_token best_token = 0;
            float best_logit = logits[0];
            for (int i = 1; i < n_vocab; i++) {
                if (logits[i] > best_logit) {
                    best_logit = logits[i];
                    best_token = i;
                }
            }

            if (llama_vocab_is_eog(g_vocab, best_token)) break;

            char piece[256];
            int piece_len = llama_token_to_piece(g_vocab, best_token, piece, sizeof(piece), 0, true);
            if (piece_len > 0) {
                result.append(piece, piece_len);
            }

            n_generated++;

            llama_token next_token = best_token;
            batch = llama_batch_get_one(&next_token, 1);
        }

        int copy_len = (int)result.size() < max_output_len - 1 ? (int)result.size() : max_output_len - 1;
        memcpy(output_buffer, result.c_str(), copy_len);
        output_buffer[copy_len] = '\0';
        return copy_len;
    } catch (...) {
        return -1;
    }
}

__declspec(dllexport) void ShutdownLLM() {
    if (g_ctx) llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    llama_backend_free();
    g_ctx = nullptr;
    g_model = nullptr;
}

}