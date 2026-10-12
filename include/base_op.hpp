#pragma once
#include <string>

#include "status.hpp"

namespace llm_bricks {

struct BaseParams {
    BaseParams() = default;
    virtual ~BaseParams() {}
};
class BaseOp {
public:
    virtual Status infer() = 0;
    void set_name(const std::string& name) {
        op_name = name;
    }

    std::string get_name() const {
        return op_name;
    }

private:
    std::string op_name;
};

}
