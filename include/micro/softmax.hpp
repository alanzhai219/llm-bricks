#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct SoftmaxParams : public BaseParams {
	SoftmaxParams(const Tensor& source, Tensor& result) : input(source), output(result) {}
	const Tensor& input;
	Tensor& output;
};

class Softmax : public BaseOp {
public:
	explicit Softmax(Context& context) : m_ctx(context) {}
	void set_input(SoftmaxParams& params) { m_params = &params; }
	Status infer() override;
private:
	Context m_ctx;
	SoftmaxParams* m_params = nullptr;
};

}  // namespace llm_bricks