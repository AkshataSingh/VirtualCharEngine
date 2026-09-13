#include <vector>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <fstream>

static Ort::Env* g_env = nullptr;
static Ort::Session* g_session = nullptr;

extern "C" {

__declspec(dllexport) bool InitTTS(const wchar_t* model_path) {
    try {
        g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "unity_tts");
        Ort::SessionOptions session_options;
        g_session = new Ort::Session(*g_env, model_path, session_options);
        return true;
    } catch (const std::exception& e) {
        std::ofstream log("C:\\VirtualCharEngine\\cpp_wrapper\\tts_error.log");
        log << "InitTTS failed: " << e.what() << std::endl;
        return false;
    }
}

__declspec(dllexport) int Synthesize(const int64_t* phoneme_ids, int num_phonemes, float* output_buffer, int max_output_samples) {
    if (!g_session) return -1;
    try {
        int64_t seq_len = num_phonemes;
        std::vector<int64_t> input_lengths = {seq_len};
        std::vector<float> scales = {0.667f, 1.0f, 0.8f};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {1, seq_len};
        std::vector<int64_t> length_shape = {1};
        std::vector<int64_t> scales_shape = {3};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(phoneme_ids), num_phonemes, input_shape.data(), input_shape.size());
        Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_lengths.data(), input_lengths.size(), length_shape.data(), length_shape.size());
        Ort::Value scales_tensor = Ort::Value::CreateTensor<float>(memory_info, scales.data(), scales.size(), scales_shape.data(), scales_shape.size());

        const char* input_names[] = {"input", "input_lengths", "scales"};
        const char* output_names[] = {"output"};

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        inputs.push_back(std::move(length_tensor));
        inputs.push_back(std::move(scales_tensor));

        auto output_tensors = g_session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

        float* audio_data = output_tensors[0].GetTensorMutableData<float>();
        size_t num_samples = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        int copy_count = static_cast<int>(num_samples < (size_t)max_output_samples ? num_samples : max_output_samples);
        memcpy(output_buffer, audio_data, copy_count * sizeof(float));
        return copy_count;
    } catch (...) {
        return -1;
    }
}

__declspec(dllexport) void ShutdownTTS() {
    delete g_session;
    delete g_env;
    g_session = nullptr;
    g_env = nullptr;
}

}
