#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <chrono>

using namespace std;

struct Pixel {
    unsigned char red, green, blue, alpha;
};

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            Pixel* p = (Pixel*)&imageRGBA[(y * width + x) * 4];

            unsigned char gray =
                (unsigned char)(p->red   * 0.2126f +
                                p->green * 0.7152f +
                                p->blue  * 0.0722f);

            p->red = p->green = p->blue = gray;
            p->alpha = 255;
        }
    }
}

int main() {

    int width, height, channels;

    auto start = chrono::high_resolution_clock::now();

    unsigned char* img = stbi_load("../ship_4k_rgba.png", &width, &height, &channels, 4);

    if (!img) {
        cerr << "ERRO: falha ao carregar imagem.\n";
        cerr << "Motivo: " << stbi_failure_reason() << endl;
        return -1;
    }

    cout << "Image loaded: " << width << " x " << height
         << " Channels original: " << channels << endl;

    // Exemplo: deixar verde
    for (int i = 0; i < width * height * 4; i += 4) {
        img[i] = 0;     // R
        img[i + 1] = 255; // G
        img[i + 2] = 0;   // B
    }

    // OU usar grayscale:
    // ConvertImageToGrayCpu(img, width, height);

    stbi_write_bmp("images/output/ship_4k_rgba_Copy.bmp",
                   width, height, 4, img);

    stbi_image_free(img);

    auto end = chrono::high_resolution_clock::now();
    cout << "Elapsed: " << chrono::duration<double>(end - start).count()
         << " seconds\n";
    
    cout << "\nPressione ENTER para sair...";
    cin.get();   // para limpar buffer
    cin.get();   // espera ENTER     

    return 0;
}