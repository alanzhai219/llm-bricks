#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct ScaleParams : public BaseParams {
	ScaleParams(const Tensor& source, float value, Tensor& result) : input(source), factor(value), output(result) {}
	const Tensor& input;
	float factor;
	Tensor& output;
};

class Scale : public BaseOp {
public:
	explicit Scale(Context& context) : m_ctx(context) {}
	void set_input(ScaleParams& params) { m_params = &params; }
	Status infer() override;
private:
	Context m_ctx;
	ScaleParams* m_params = nullptr;
};

}  // namespace llm_bricks