#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

// Write raw float audio samples to a 16-bit PCM WAV file
void write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    std::ofstream f(path, std::ios::binary);

    int32_t data_size = static_cast<int32_t>(samples.size() * sizeof(int16_t));
    int32_t chunk_size = 36 + data_size;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * sizeof(int16_t);
    int16_t block_align = num_channels * sizeof(int16_t);
    int16_t bits_per_sample = 16;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    int16_t audio_format = 1;
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&num_channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);

    for (float s : samples) {
        int16_t sample = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
        f.write(reinterpret_cast<const char*>(&sample), 2);
    }
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tts_wrapper");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"en_US-lessac-medium.onnx", session_options);

    // Phoneme IDs for "Hello there." (from Python's real phonemization)
    std::vector<int64_t> phoneme_ids = {1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 3, 0, 41, 0, 120, 0, 61, 0, 88, 0, 10, 0, 2};
    int64_t seq_len = static_cast<int64_t>(phoneme_ids.size());
    std::vector<int64_t> input_lengths = {seq_len};
    std::vector<float> scales = {0.667f, 1.0f, 0.8f};  // noise_scale, length_scale, noise_scale_w

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, seq_len};
    std::vector<int64_t> length_shape = {1};
    std::vector<int64_t> scales_shape = {3};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, phoneme_ids.data(), phoneme_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_lengths.data(), input_lengths.size(), length_shape.data(), length_shape.size());
    Ort::Value scales_tensor = Ort::Value::CreateTensor<float>(memory_info, scales.data(), scales.size(), scales_shape.data(), scales_shape.size());

    const char* input_names[] = {"input", "input_lengths", "scales"};
    const char* output_names[] = {"output"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    inputs.push_back(std::move(length_tensor));
    inputs.push_back(std::move(scales_tensor));

    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

    float* audio_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_samples = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Generated " << num_samples << " audio samples." << std::endl;

    std::vector<float> samples(audio_data, audio_data + num_samples);
    write_wav("cpp_output.wav", samples, 22050);
    std::cout << "Saved to cpp_output.wav" << std::endl;

    return 0;
}