#pragma once

#include <vector>
#include <string>

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <cstring>

#include "device.hpp"
#include "dtype.hpp"

namespace llm_bricks {

static size_t compute_numel(const std::vector<size_t>& shape) {
    size_t numel = 1;
    for (size_t dim : shape) {
        numel *= dim;
    }
    return numel;
}

struct Tensor {
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::f32) : m_shape(shape), m_dtype(dtype) {
        compute_strides();
        switch (m_dtype) {
            case DType::f32:
                m_data = malloc(compute_numel(m_shape) * sizeof(float));
                break;
            /*
            case DType::f16:
                m_data.resize(numel() * sizeof(uint16_t));
                break;
            case DType::bf16:
                m_data.resize(numel() * sizeof(uint16_t));
                break;
            case DType::i8:
                m_data.resize(numel() * sizeof(int8_t));
                break;
            case DType::u8:
                m_data.resize(numel() * sizeof(uint8_t));
                break;
            case DType::unknown:
            */
            default:
                throw std::runtime_error("Tensor dtype must be specified");
        }
    }
    virtual ~Tensor() {
        free(m_data);
    }
    // Tensor(const std::vector<size_t>& shape, T fill = T(0));
    Tensor(const Tensor& other) {
        m_shape = other.m_shape;
        m_stride = other.m_stride;
        m_dtype = other.m_dtype;
        m_data = malloc(compute_numel(m_shape) * sizeof(float));
        std::memcpy(m_data, other.m_data, compute_numel(m_shape) * sizeof(float));
    }
    Tensor(Tensor&& other) {
        m_shape = std::move(other.m_shape);
        m_stride = std::move(other.m_stride);
        m_dtype = other.m_dtype;
        m_data = other.m_data;
        other.m_data = nullptr;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            free(m_data);
            m_shape = other.m_shape;
            m_stride = other.m_stride;
            m_dtype = other.m_dtype;
            m_data = malloc(compute_numel(m_shape) * sizeof(float));
            std::memcpy(m_data, other.m_data, compute_numel(m_shape) * sizeof(float));
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) {
        if (this != &other) {
            free(m_data);
            m_shape = std::move(other.m_shape);
            m_stride = std::move(other.m_stride);
            m_dtype = other.m_dtype;
            m_data = other.m_data;
            other.m_data = nullptr;
        }
        return *this;
    }
    
    // shape: [2,3,4] => stride [12,4,1]
    // dim:   n-1, n-2, ..., 1, 0
    // idx of vec: 0, 1, 2, ..., n-2, n-1
    void compute_strides() {
        m_stride.resize(m_shape.size()); 
        m_stride.back() = 1;
        for (size_t idx = m_shape.size() - 1; idx > 0; --idx) {
            m_stride[idx - 1] = m_stride[idx] * m_shape[idx];
        }
    }

    void reshape(const std::vector<size_t>& shape) {
        assert(numel() == compute_numel(shape));
        m_shape = shape;
        compute_strides();
    }

    // access raw data
    void* ptr() {
        return m_data;
    }

    const void* ptr() const {
        return m_data;
    }

    // access tensor data by index
    void* ptr(size_t i, size_t j, size_t k, size_t l) {
        assert(m_shape.size() == 4);
        const size_t offset = i * m_stride[0] + j * m_stride[1] + k * m_stride[2] + l * m_stride[3];
        return static_cast<uint8_t*>(m_data) + offset;
    }

    void* ptr(size_t i, size_t j, size_t k) {
        assert(m_shape.size() == 3);
        const size_t offset = i * m_stride[0] + j * m_stride[1] + k * m_stride[2];
        return static_cast<uint8_t*>(m_data) + offset;
    }

    void* ptr(size_t i, size_t j) {
        assert(m_shape.size() == 2);
        const size_t offset = i * m_stride[0] + j * m_stride[1];
        return static_cast<uint8_t*>(m_data) + offset; 
    }

    // access shape
    const std::vector<size_t>& stride() const {
        return m_stride;
    }

    const std::vector<size_t>& shape() const {
        return m_shape;
    }

    const DType dtype() const {
        return m_dtype;
    }

    const size_t ndims() const {
        return m_shape.size();
    }

    const size_t dim(size_t idx) const {
        assert(idx < m_shape.size());
        return m_shape[idx];
    }

    std::string shape_str() const {
        std::string str = "[";
        for (size_t i = 0; i < m_shape.size(); ++i) {
            str += std::to_string(m_shape[i]);
            if (i != m_shape.size() - 1) {
                str += ",";
            }
        }
        str += "]";
        return str;
    }

    size_t numel() const {
        size_t num = 1;
        for (auto v : m_shape) {
            num *= v;
        }
        return num;
    }

private:
  void* m_data;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_stride;
  DType m_dtype = DType::unknown;
};

} // namespace llm_bricks
