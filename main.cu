#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <functional>
#include <cstdlib>
#include <cstring>

using namespace std;

struct Image {
    int width = 0, height = 0, channels = 0;
    unsigned char* data = nullptr;

    Image(const string& filename) {
        data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
        if (!data) width = 0;
        else channels = 3;
    }

    Image(const Image& other) {
        width = other.width;
        height = other.height;
        channels = other.channels;
        size_t size = width * height * channels;
        if (other.data) {
            data = (unsigned char*)malloc(size);
            memcpy(data, other.data, size);
        }
    }

    ~Image() {
        if (data) stbi_image_free(data);
    }
    
    bool isValid() const { return data != nullptr; }
};

void filter_green_cpu(unsigned char* p) {
    p[0] = 0;
    p[2] = 0;
}

void filter_invert_cpu(unsigned char* p) {
    p[0] = 255 - p[0];
    p[1] = 255 - p[1];
    p[2] = 255 - p[2];
}

void filter_grayscale_cpu(unsigned char* p) {
    unsigned char gray = (p[0] + p[1] + p[2]) / 3;
    p[0] = gray;
    p[1] = gray;
    p[2] = gray;
}

void engine_sequential(Image& img, int filterType) {
    size_t total = (size_t)img.width * img.height;
    unsigned char* p = img.data;
    for(size_t i = 0; i < total; ++i) {
        if(filterType == 1) filter_green_cpu(p);
        else if(filterType == 2) filter_invert_cpu(p);
        else if(filterType == 3) filter_grayscale_cpu(p);
        p += 3;
    }
}

void worker(unsigned char* start, size_t count, int filterType) {
    unsigned char* p = start;
    for(size_t i = 0; i < count; ++i) {
        if(filterType == 1) filter_green_cpu(p);
        else if(filterType == 2) filter_invert_cpu(p);
        else if(filterType == 3) filter_grayscale_cpu(p);
        p += 3;
    }
}

void engine_multithread(Image& img, int filterType) {
    unsigned int num_threads = thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;

    vector<thread> threads;
    size_t total = (size_t)img.width * img.height;
    size_t chunk = total / num_threads;
    unsigned char* ptr = img.data;

    for(unsigned int i = 0; i < num_threads; ++i) {
        size_t size = (i == num_threads - 1) ? (total - i * chunk) : chunk;
        threads.emplace_back(worker, ptr, size, filterType);
        ptr += size * 3;
    }

    for(auto& t : threads) t.join();
}

__global__ void kernel_green(unsigned char* img, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        int idx = i * 3;
        img[idx] = 0;
        img[idx+2] = 0;
    }
}

__global__ void kernel_invert(unsigned char* img, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        int idx = i * 3;
        img[idx] = 255 - img[idx];
        img[idx+1] = 255 - img[idx+1];
        img[idx+2] = 255 - img[idx+2];
    }
}

__global__ void kernel_grayscale(unsigned char* img, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        int idx = i * 3;
        unsigned char gray = (img[idx] + img[idx+1] + img[idx+2]) / 3;
        img[idx] = gray;
        img[idx+1] = gray;
        img[idx+2] = gray;
    }
}

void engine_cuda(Image& img, int filterType) {
    unsigned char* d_img = nullptr;
    int total_pixels = img.width * img.height;
    size_t dataSize = total_pixels * 3 * sizeof(unsigned char);

    cudaMalloc(&d_img, dataSize);
    cudaMemcpy(d_img, img.data, dataSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (total_pixels + blockSize - 1) / blockSize;

    if (filterType == 1) kernel_green<<<numBlocks, blockSize>>>(d_img, total_pixels);
    else if (filterType == 2) kernel_invert<<<numBlocks, blockSize>>>(d_img, total_pixels);
    else if (filterType == 3) kernel_grayscale<<<numBlocks, blockSize>>>(d_img, total_pixels);

    cudaDeviceSynchronize();
    cudaMemcpy(img.data, d_img, dataSize, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}

template<class Func, class... Args>
double measure_time(Func&& func, Args&&... args) {
    auto t0 = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double, std::milli>(t1 - t0).count();
}

void show_image_windows(const string& filename) {
    string cmd = "start \"\" \"" + filename + "\"";
    system(cmd.c_str());
}

int main() {
    string inputPath;
    cout << "=== PROJETO 01: PAVIC LAB 2025 (COM CUDA) ===\n";
    cout << "Digite o nome da imagem:\n> ";
    
    getline(cin, inputPath);
    if (!inputPath.empty() && inputPath.front() == '"') inputPath.erase(0, 1);
    if (!inputPath.empty() && inputPath.back() == '"') inputPath.pop_back();
    if (inputPath.empty()) return 0;

    Image original(inputPath);
    if (!original.isValid()) {
        cout << "Erro ao abrir imagem.\n";
        return 1;
    }

    cout << "Imagem carregada: " << original.width << "x" << original.height << endl;
    show_image_windows(inputPath);

    while (true) {
        cout << "\n--- MENU DE CONTROLE ---\n";
        cout << "1. Filtro: Tons de Verde\n";
        cout << "2. Filtro: Inverter Cores\n";
        cout << "3. Filtro: Escala de Cinza\n";
        cout << "0. Sair\n";
        cout << "Escolha o filtro: ";
        
        int opFiltro;
        if (!(cin >> opFiltro)) {
            cin.clear(); cin.ignore(10000, '\n'); continue; 
        }
        if (opFiltro == 0) break;

        string filterName;
        if(opFiltro == 1) filterName = "verde";
        else if(opFiltro == 2) filterName = "invert";
        else if(opFiltro == 3) filterName = "gray";
        else continue;

        cout << "\n--- MODO DE PROCESSAMENTO ---\n";
        cout << "1. Sequencial (CPU)\n";
        cout << "2. Multithread (CPU)\n";
        cout << "3. CUDA (GPU NVIDIA)\n";
        cout << "Escolha o modo: ";

        int opMode;
        cin >> opMode;

        Image workingImg(original); 
        double timeTaken = 0.0;
        string modeName;

        if (opMode == 1) {
            modeName = "Seq";
            timeTaken = measure_time(engine_sequential, ref(workingImg), opFiltro);
        } else if (opMode == 2) {
            modeName = "Multi";
            timeTaken = measure_time(engine_multithread, ref(workingImg), opFiltro);
        } else if (opMode == 3) {
            modeName = "CUDA";
            timeTaken = measure_time(engine_cuda, ref(workingImg), opFiltro);
        } else {
            cout << "Modo invalido!\n";
            continue;
        }

        cout << "\n>>> RESULTADO <<<\n";
        cout << "Filtro: " << filterName << " | Modo: " << modeName << endl;
        cout << "Tempo: " << timeTaken << " ms" << endl;

        string outName = "saida_" + filterName + "_" + modeName + ".jpg";
        stbi_write_jpg(outName.c_str(), workingImg.width, workingImg.height, 3, workingImg.data, 100);
        
        cout << "Salvo: " << outName << endl;
        show_image_windows(outName);
    }

    return 0;
}