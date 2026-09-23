#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct GeluParams : public BaseParams {
	explicit GeluParams(Tensor& source) : input(source) {}
	Tensor& input;
};

class Gelu : public BaseOp {
public:
	explicit Gelu(Context& context) : m_ctx(context) {}
	void set_input(GeluParams& params) { m_params = &params; }
	Status infer() override;
private:
	Context m_ctx;
	GeluParams* m_params = nullptr;
};

}  // namespace llm_bricks