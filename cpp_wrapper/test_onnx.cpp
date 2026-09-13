#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    std::cout << "ONNX Runtime environment created successfully!" << std::endl;
    return 0;
}