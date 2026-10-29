# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

# Helpers shared by the tests and the benchmarks. They need only JuMP.

module TestCommon

using JuMP

_nonzeros(f::AffExpr) = length(f.terms)
_nonzeros(f::AbstractVector{AffExpr}) = sum(_nonzeros, f; init = 0)
_nonzeros(::Any) = 0  # bounds and other sets of variables

# Variables, constraints (without the bounds of variables) and nonzeros of the affine constraints,
# scalar or vector.
function model_size(model::Model)
    nonzeros = 0
    for (F, S) in list_of_constraint_types(model)
        for c in all_constraints(model, F, S)
            nonzeros += _nonzeros(constraint_object(c).func)
        end
    end
    constraints = num_constraints(model; count_variable_in_set_constraints = false)
    return (variables = num_variables(model), constraints = constraints, nonzeros = nonzeros)
end

end # module TestCommon
