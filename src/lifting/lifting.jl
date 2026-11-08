# Copyright (c) 2024: Joaquim Dias Garcia, Bernardo Freitas Paulo da Costa, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""
    AbstractLifting

A lifting of some uncertainties `η` into more coordinates `η̃ = L(η)` (Georghiou, Wiesemann and
Kuhn, 2015): a decision rule that is linear in `η̃` is nonlinear in `η`.

A lifting `l` implements:

 * [`lift`](@ref)`(l, η)`, the nonlinear map `L`, applied to the outcomes;
 * [`retraction`](@ref)`(l)`, the sparse matrix `R` of the linear retraction, with `R L(η) = η`
   for a lifting that replaces its inputs; the data of the model are lifted with it (`B ← BR`,
   `C ← CR`, `W ← WR`).

A lifted coordinate is in the largest stage of the uncertainties that it depends on.

There are two liftings in 0.4. [`AxisPiecewise`](@ref), the lifting of [`BreakPoints`](@ref),
replaces its one uncertainty by its pieces. [`CombinationPiecewise`](@ref), the lifting of a
combination of several uncertainties (see [`Experimental.set_breakpoints`](@ref)), keeps its
inputs as coordinates and adds the pieces of the combination: its retraction is zero, since the
inputs are read from their own columns. Subtyping `AbstractLifting` is not supported yet: a lifting
also implements internal methods (its sizes, the bounds and hull of its coordinates, its moments),
whose contract may change until the `Experimental` submodule.
"""
abstract type AbstractLifting end

"""
    lift(l::AbstractLifting, η::AbstractVector{<:Real}) -> Vector{Float64}
    lift(l::AbstractLifting, η::Real) -> Vector{Float64}

Return the lifted coordinates `L(η)` of the outcome `η` of the uncertainties of `l`; a lifting of
one uncertainty also takes a scalar.
"""
function lift end

lift(l::AbstractLifting, η::Real) = lift(l, [η])

"""
    retraction(l::AbstractLifting) -> SparseMatrixCSC{Float64,Int}

Return the matrix `R` of the linear retraction of `l`, one row per uncertainty and one column per
lifted coordinate, such that `R * lift(l, η) == η` for a lifting that replaces its inputs, such as
[`AxisPiecewise`](@ref). A lifting that keeps its inputs as coordinates, such as
[`CombinationPiecewise`](@ref), has a zero retraction: its inputs are read from their own
columns.
"""
function retraction end

"""
    LiftingBlock(lifting::AbstractLifting, inputs::Vector{Int}, columns::Vector{Int})

One lifting of a [`LiftingMap`](@ref): the columns `columns` of the lifted `ξ̃` are
`lift(lifting, ξ[inputs])`.

## Fields

 * `lifting::AbstractLifting`
 * `inputs::Vector{Int}`: the columns of the original `ξ` that it lifts.
 * `columns::Vector{Int}`: the columns of `ξ̃` of the lifted coordinates, in the order of
   `lift(lifting, ξ[inputs])`.
"""
struct LiftingBlock
    lifting::AbstractLifting
    inputs::Vector{Int}
    columns::Vector{Int}
end

"""
    LiftingMap

The map between the outcomes `ξ = (1, η)` of the uncertainties of a model and the lifted
coordinates `ξ̃ = (1, η̃)` in which its rules are written: [`rule_matrix`](@ref) returns it in the
field `lifting` of a [`RuleMatrix`](@ref). Internally, [`lift`](@ref)`(form, liftings)` builds
it, and keeps it in the field `lifting` of the lifted form.

Apply the lifting `L` to an outcome of the original uncertainties with `lift(map, ξ)`, and the
retraction with `retraction(map) * ξ̃`. A decision rule `x(ξ̃) = X̃ ξ̃` of the lifted form is the
rule `x(ξ) = X̃ lift(map, ξ)` on the original uncertainties.

## Fields

 * `uncertainties::Vector{MOI.VariableIndex}`: the original uncertainties, in the columns
   `2:1+m` of `ξ`.
 * `uncertainty_names::Vector{String}`: the name of each original uncertainty.
 * `uncertainty_stages::Vector{Int}`: the stage of each original uncertainty.
 * `kept::Vector{Int}`: the column of `ξ̃` of each column of `ξ` that is not lifted (the constant
   column 1 is column 1), and 0 for a lifted column.
 * `blocks::Vector{LiftingBlock}`: the liftings, each with its columns in `ξ` and in `ξ̃`.
 * `R::SparseMatrixCSC{Float64,Int}`: the retraction, `(1 + m) × (1 + m̃)`, with `ξ = R ξ̃`,
   assembled from the identity on the kept columns and the [`retraction`](@ref) of each block.
"""
struct LiftingMap
    uncertainties::Vector{MOI.VariableIndex}
    uncertainty_names::Vector{String}
    uncertainty_stages::Vector{Int}
    kept::Vector{Int}
    blocks::Vector{LiftingBlock}
    R::SparseArrays.SparseMatrixCSC{Float64,Int}
end

retraction(map::LiftingMap) = map.R

# Whether the lifted coordinates come in addition to the inputs, which then stay columns of ξ̃
# (`CombinationPiecewise`), or replace them (`AxisPiecewise`).
_keeps_inputs(::AbstractLifting) = false

# The column of ξ̃ of the column `j` of ξ, or 0 if it is lifted.
_column_of(map::LiftingMap, j::Int) = map.kept[j]

# The block that replaces the column `j` of ξ by its lifted coordinates.
function _block_of(map::LiftingMap, j::Int)
    return only(b for b in map.blocks if !_keeps_inputs(b.lifting) && j in b.inputs)
end

# The column of ξ that `block` lifts, for a lifting of one uncertainty (`AxisPiecewise`): the rules
# and the results read the input of such a block only here.
_input(block::LiftingBlock) = only(block.inputs)

# The label of each column of ξ̃, from the label `labels[j - 1]` of each column `j` of ξ: `:constant`,
# the label of a column that is not lifted, `(label, k)` for the `k`-th lifted coordinate of a
# column, and `((label₁, label₂, …), k)` for the `k`-th piece of a combination of columns, whose
# pieces are numbered from 2 (see `CombinationPiecewise`).
function _column_labels(map::LiftingMap, labels::AbstractVector)
    columns = Vector{Any}(undef, size(map.R, 2))
    columns[1] = :constant
    for j in 2:length(map.kept)
        c = _column_of(map, j)
        c > 0 && (columns[c] = labels[j-1])
    end
    for block in map.blocks, (k, c) in enumerate(block.columns)
        columns[c] = if _keeps_inputs(block.lifting)
            (Tuple(labels[j-1] for j in block.inputs), k + 1)
        else
            (labels[_input(block)-1], k)
        end
    end
    return columns
end

"""
    lift(map::LiftingMap, ξ::AbstractVector{<:Real}) -> Vector{Float64}
    lift(map::LiftingMap, ξ::AbstractMatrix{<:Real}) -> Matrix{Float64}

Return the lifted outcome `ξ̃ = L(ξ)` of the outcome `ξ = (1, η)` of the original uncertainties,
in the columns of the lifted form; with a matrix, each column is an outcome.
"""
function lift(map::LiftingMap, ξ::AbstractVector{<:Real})
    m, m̃ = size(map.R)
    if length(ξ) != m
        throw(DimensionMismatch("The outcome has $(length(ξ)) entries; expected $m, ξ = (1, η)."))
    end
    ξ̃ = zeros(m̃)
    for (j, k) in enumerate(map.kept)
        k > 0 && (ξ̃[k] = ξ[j])
    end
    for block in map.blocks
        ξ̃[block.columns] = lift(block.lifting, view(ξ, block.inputs))
    end
    return ξ̃
end

function lift(map::LiftingMap, ξ::AbstractMatrix{<:Real})
    ξ̃ = zeros(size(map.R, 2), size(ξ, 2))
    for (s, outcome) in enumerate(eachcol(ξ))
        ξ̃[:, s] = lift(map, outcome)
    end
    return ξ̃
end
