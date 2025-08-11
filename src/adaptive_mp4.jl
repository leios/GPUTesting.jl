# defining a precision wrapper for the hierarchical data structure
# AbstractBlock -> MatrixBlock -> HierarchicalBlock
abstract type AbstractBlock end

struct MatrixBlock{T} <: AbstractBlock
    data::Matrix{T}
end

struct HierarchicalBlock <: AbstractBlock
    blocks::NTuple{4, AbstractBlock}
end







# recursive matrix builder
"""
    build_triangular_matrix(matrix::Matrix{Float64}, precisions::Vector{DataType}, is_lower::Bool) -> AbstractBlock

Recursively constructs a hierarchical mixed-precision triangular matrix from a full square matrix.

The matrix is subdivided into 4 subblocks:
- Diagonal blocks (`A11` and `A22`) are recursively split,
- Off-diagonal blocks are stored at the current precision level,
- Off-triangle blocks are set to `nothing` to respect triangular structure.

### Arguments
- matrix::Matrix{Float64}: A square matrix to be wrapped into a hierarchical structure.
- precisions::Vector{DataType}: Array of precisions (e.g., [Float16, Float32, Float64]), where:
    - precisions[1] is used for off-diagonal blocks at the current level,
    - deeper levels use precisions[2:end] for diagonals.
- is_lower::Bool: If true, constructs a lower triangular matrix; if false, constructs upper triangular.

### Returns
- AbstractBlock: A HierarchicalBlockTriangular object that stores the recursively structured triangular matrix,
  with off-diagonal blocks cast to the appropriate precision and triangle structure respected.

### Example Usage
A = rand(8, 8)
precisions = [Float16, Float32, Float64]
A_hier = build_triangular_matrix(A, precisions, true)  # Lower-triangular hierarchy
"""
function build_hierarchical_matrix(matrix::Matrix{Float64}, precisions::Vector{DataType})::AbstractBlock
    n = size(matrix, 1)
    if length(precisions) == 0 || n == 1
        return MatrixBlock{Float64}(matrix)
    end

    mid = div(n, 2)
    A11 = matrix[1:mid, 1:mid]
    A12 = matrix[1:mid, mid+1:end]
    A21 = matrix[mid+1:end, 1:mid]
    A22 = matrix[mid+1:end, mid+1:end]

    T = precisions[1]

    block11 = build_hierarchical_matrix(A11, precisions[2:end])
    block22 = build_hierarchical_matrix(A22, precisions[2:end])
    block12 = MatrixBlock{T}(convert(Matrix{T}, A12))
    block21 = MatrixBlock{T}(convert(Matrix{T}, A21))

    return HierarchicalBlock((block11, block12, block21, block22))
end



#  Adding triangular matrix support to the recursive matrix builder
function build_triangular_matrix(
    matrix::Matrix{Float64},
    precisions::Vector{DataType},
    is_lower::Bool
)::AbstractBlock
    n = size(matrix, 1)
    if length(precisions) == 0 || n == 1
        return MatrixBlock{Float64}(matrix)
    end

    mid = div(n, 2)
    A11 = matrix[1:mid, 1:mid]
    A12 = matrix[1:mid, mid+1:end]
    A21 = matrix[mid+1:end, 1:mid]
    A22 = matrix[mid+1:end, mid+1:end]

    T = precisions[1]

    block11 = build_triangular_matrix(A11, precisions[2:end], is_lower)
    block22 = build_triangular_matrix(A22, precisions[2:end], is_lower)

    block12 = is_lower ? nothing : MatrixBlock{T}(convert(Matrix{T}, A12))
    block21 = is_lower ? MatrixBlock{T}(convert(Matrix{T}, A21)) : nothing

    return HierarchicalBlockTriangular((block11, block12, block21, block22))
end

# TRMM with Lower Triangular Matrix
function multiply_triangular_dense(
    A::HierarchicalBlockTriangular,
    B::Matrix,
    is_lower::Bool
)::Matrix
    n = size(B, 1)
    mid = div(n, 2)
    B1 = B[1:mid, :]
    B2 = B[mid+1:end, :]

    A11, A12, A21, A22 = A.blocks

    if is_lower
        # C1 = A11 * B1
        C1 = multiply_triangular_dense(A11, B1, true)

        # C2 = A21 * B1 + A22 * B2
        C21 = A21 !== nothing ? multiply_triangular_dense(A21, B1, true) : zeros(size(B2,1), size(B,2))
        C22 = multiply_triangular_dense(A22, B2, true)
        C2 = C21 + C22
    else
        # Upper triangular
        # C1 = A11 * B1 + A12 * B2
        C11 = multiply_triangular_dense(A11, B1, false)
        C12 = A12 !== nothing ? multiply_triangular_dense(A12, B2, false) : zeros(size(B1,1), size(B,2))
        C1 = C11 + C12

        # C2 = A22 * B2
        C2 = multiply_triangular_dense(A22, B2, false)
    end

    return vcat(C1, C2)
end