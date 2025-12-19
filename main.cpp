#define _CRT_SECURE_NO_WARNINGS
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

// Estrutura da Imagem
struct Image {
    int width = 0, height = 0, channels = 0;
    unsigned char* data = nullptr;

    Image(const string& filename) {
        data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
        if (!data) {
            width = 0; 
        } else {
            channels = 3;
        }
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

// Filtros
void filter_green(unsigned char* p) {
    p[0] = 0;
    p[2] = 0;
}

void filter_invert(unsigned char* p) {
    p[0] = 255 - p[0];
    p[1] = 255 - p[1];
    p[2] = 255 - p[2];
}

void filter_grayscale(unsigned char* p) {
    unsigned char gray = (p[0] + p[1] + p[2]) / 3;
    p[0] = gray;
    p[1] = gray;
    p[2] = gray;
}

// Motores de Processamento
void engine_sequential(Image& img, function<void(unsigned char*)> filter) {
    size_t total = (size_t)img.width * img.height;
    unsigned char* p = img.data;
    for(size_t i = 0; i < total; ++i) {
        filter(p);
        p += 3;
    }
}

void worker(unsigned char* start, size_t count, function<void(unsigned char*)> filter) {
    unsigned char* p = start;
    for(size_t i = 0; i < count; ++i) {
        filter(p);
        p += 3;
    }
}

void engine_multithread(Image& img, function<void(unsigned char*)> filter) {
    unsigned int num_threads = thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;

    vector<thread> threads;
    size_t total = (size_t)img.width * img.height;
    size_t chunk = total / num_threads;
    unsigned char* ptr = img.data;

    for(unsigned int i = 0; i < num_threads; ++i) {
        size_t size = (i == num_threads - 1) ? (total - i * chunk) : chunk;
        threads.emplace_back(worker, ptr, size, filter);
        ptr += size * 3;
    }

    for(auto& t : threads) t.join();
}

void engine_cuda_simulation(Image& img) {
    this_thread::sleep_for(chrono::milliseconds(50));
}

template<class Func, class... Args>
double measure_time(Func&& func, Args&&... args) {
    auto t0 = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double, std::milli>(t1 - t0).count();
}

void show_image_windows(const string& filename) {
    string cmd = "start \"\" \"" + filename + "\""; // Aspas extras para caminhos com espaço
    system(cmd.c_str());
}

int main() {
    string inputPath;
    cout << "=== PROJETO 01: PAVIC LAB 2025 ===\n";
    cout << "Digite o nome ou caminho da imagem:\n> ";
    
    // CORREÇÃO AQUI: Usar getline para aceitar espaços
    getline(cin, inputPath);

    // Remove aspas se o usuário tiver copiado como "Caminho"
    if (!inputPath.empty() && inputPath.front() == '"') inputPath.erase(0, 1);
    if (!inputPath.empty() && inputPath.back() == '"') inputPath.pop_back();

    if (inputPath.empty()) return 0;

    Image original(inputPath);
    if (!original.isValid()) {
        cout << "Erro ao abrir imagem: [" << inputPath << "]\n";
        cout << "Verifique se o caminho esta correto e sem aspas extras.\n";
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
        if (!(cin >> opFiltro)) { // Proteção contra letras
            cin.clear(); 
            cin.ignore(10000, '\n'); 
            continue; 
        }
        if (opFiltro == 0) break;

        function<void(unsigned char*)> selectedFilter;
        string filterName;

        switch(opFiltro) {
            case 1: selectedFilter = filter_green; filterName = "verde"; break;
            case 2: selectedFilter = filter_invert; filterName = "invert"; break;
            case 3: selectedFilter = filter_grayscale; filterName = "gray"; break;
            default: cout << "Opcao invalida!\n"; continue;
        }

        cout << "\n--- MODO DE PROCESSAMENTO ---\n";
        cout << "1. Sequencial (CPU 1 Core)\n";
        cout << "2. Multithread (CPU Multi Core)\n";
        cout << "3. CUDA (GPU - Simulado)\n";
        cout << "Escolha o modo: ";

        int opMode;
        cin >> opMode;

        Image workingImg(original); 
        double timeTaken = 0.0;
        string modeName;

        if (opMode == 1) {
            modeName = "Seq";
            timeTaken = measure_time(engine_sequential, ref(workingImg), selectedFilter);
        } else if (opMode == 2) {
            modeName = "Multi";
            timeTaken = measure_time(engine_multithread, ref(workingImg), selectedFilter);
        } else if (opMode == 3) {
            modeName = "CUDA";
            timeTaken = measure_time(engine_cuda_simulation, ref(workingImg));
        } else {
            cout << "Modo invalido!\n";
            continue;
        }

        cout << "\n>>> RESULTADO <<<\n";
        cout << "Filtro: " << filterName << " | Modo: " << modeName << endl;
        cout << "Tempo de Execucao: " << timeTaken << " ms" << endl;

        string outName = "saida_" + filterName + "_" + modeName + ".jpg";
        stbi_write_jpg(outName.c_str(), workingImg.width, workingImg.height, 3, workingImg.data, 100);
        
        cout << "Imagem salva: " << outName << endl;
        show_image_windows(outName);
    }

    return 0;
}