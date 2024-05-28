#include <iostream>

int main(int argc, char **argv) {
    const size_t image_width = 256; 
    const size_t image_height = 256; 

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n"; 
    for (size_t row = 0; row < image_height; ++row) {
        for (size_t col = 0; col < image_width; ++col) {
            const double r = double(row) / (image_height - 1); 
            const double g = double(col) / (image_width - 1); 
            const double b = 0; 

            const int r_byte = static_cast<int>(255.999 * r); 
            const int g_byte = static_cast<int>(255.999 * g); 
            const int b_byte = static_cast<int>(255.999 * b); 

            std::cout << r_byte << ' ' << g_byte << ' ' << b_byte << '\n'; 
        }
    }
}