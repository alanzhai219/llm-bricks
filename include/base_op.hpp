#pragma once
#include "status.hpp"

namespace llm_bricks {

struct BaseParams {
    BaseParams() = default;
    virtual ~BaseParams() {}
};
class BaseOp {
public:
    virtual Status infer() = 0;
};

}
