#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct AddParams : public BaseParams {
	AddParams(const Tensor& left, const Tensor& right, Tensor& result) : lhs(left), rhs(right), output(result) {}
	const Tensor& lhs;
	const Tensor& rhs;
	Tensor& output;
};

class Add : public BaseOp {
public:
	explicit Add(Context& context) : m_ctx(context) {}
	void set_input(AddParams& params) { m_params = &params; }
	Status infer() override;
private:
	Context m_ctx;
	AddParams* m_params = nullptr;
};

}  // namespace llm_bricks