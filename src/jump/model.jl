# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    LDRModel(inner_factory = nothing)

A `JuMP.Model` whose optimizer is [`Optimizer`](@ref)`(inner_factory)`, without bridges.

`inner_factory`, for example `HiGHS.Optimizer`, builds the optimizer that solves the
reformulated models. Anything the optimizer does not read, such as a second-order cone, is
rejected when it is added to the model.

## Example

```julia
model = LDRModel(HiGHS.Optimizer)
@variable(model, buy >= 0, FirstStage)
@variable(model, d in Uncertainty(Uniform(80, 120)))
```
"""
function LDRModel(inner_factory = nothing)
    return JuMP.Model(() -> Optimizer(inner_factory); add_bridges = false)
end
