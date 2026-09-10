#include <vector>

template <typename T>
struct Tensor4D {
public:
    Tensor4D(const std::vector<size_t>& shape)
        : B(shape[0]), 
          H(shape[1]), 
          S(shape[2]),
          D(shape[3]),
          data(B * H * S * D)
    {}

    T* ptr() {
        return data.data();
    }

    // [B,H,S,D]
    T& at(size_t b, size_t h, size_t s, size_t d) {
        size_t offset = ((b * H + h) * S + s) * D + d;

        return data[offset];
    }

    const T& at(size_t b, size_t h, size_t s, size_t d) const {
        size_t offset = ((b * H + h) * S + s) * D + d;

        return data[offset];
    }

    const std::vector<size_t> get_shape() const {
        return shape;
    }


    std::vector<size_t> get_shape() {
        return shape;
    }

    void dump() {
        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                std::cout << "B=" << b << ", H=" << h << "\n";

                for (size_t s = 0; s < S; ++s) {
                    std::cout << "  S=" << s << ": [ ";
                    for (size_t d = 0; d < D; ++d) {
                        std::cout << std::fixed << std::setprecision(4) << x.at(b,h,s,d) << " ";
                    }
                    std::cout << "]\n";
                }
                std::cout << "\n";
            }
        }
    }

private:
    size_t B;
    size_t H;
    size_t S;
    size_t D;

    std::vector<T> data;
    std::vector<size_t> shape;
};

static void print_tensor()
