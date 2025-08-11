#export TriangularBlockMatrix, build_block_structure, fast_get

struct TriangularBlockMatrix{T}
    blocks::Vector{Vector{Matrix{T}}}
    n::Int
    l::Int
    triangular_type::Symbol  # :lower or :upper


    Base.iterate(::TriangularBlockMatrix) = nothing
    Base.iterate(::TriangularBlockMatrix, state) = nothing
end

"""
Constructs a TriangularBlockMatrix from a square matrix 'A'
and a given number of levels 'l' (or inferred from 'U_approx')
"""
function TriangularBlockMatrix(A::Matrix{T}, U_approx; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    l = size(U_approx, 1)
    blocks = build_block_structure(A, l; triangular_type)
    return TriangularBlockMatrix{T}(blocks, n, l, triangular_type)
end




"""
build_block_structure: computes and returns the block structure of a matrix

inputs: A: the square matrix (will be specialized for triangular matrices)
       
        l: the number of levels of the hierarchical block structure

        triangular_type::Symbol: upper or lower to indicate the structure of the
                                triangular matrix

output: blocks: format Vector{Vector{Matrix{T}}} where 
                    blocks is an Array of the individual arrays for each level
                    blocks[i] is the Array of blocks for a particular level i
                    blocks[i][k] is the matrix of elements in a block k of level i


Modified block builder with triangular handling
"""
# function build_block_structure(A, l; triangular_type::Symbol = :lower)
#     n = size(A, 1)
#     blocks = [Matrix[] for _ in 1:l+1]

#     function subdivide(A, level)
#         if level == l+1
#             push!(blocks[level], A)
#             return
#         end

#         mid = cld(size(A, 1), 2)
#         top, bottom = 1:mid, (mid + 1):size(A, 1)
#         left, right = 1:mid, (mid + 1):size(A, 2)

#         A11 = A[top, left]
#         A12 = A[top, right]
#         A21 = A[bottom, left]
#         A22 = A[bottom, right]

#         if triangular_type == :lower
#             push!(blocks[level], A21)
#         elseif triangular_type == :upper
#             push!(blocks[level], A12)
#         else
#             error("Unsupported triangular_type. Use :lower or :upper.")
#         end

#         subdivide(A11, level + 1)
#         subdivide(A22, level + 1)
#     end

#     subdivide(A, 1)
#     return blocks
# end

function build_block_structure(A::AbstractMatrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    blocks = [Matrix{T}[] for _ in 1:l+1]

    function subdivide(i_start::Int, j_start::Int, size::Int, level::Int)
        if level == l + 1
            view_block = @view A[i_start:i_start+size-1, j_start:j_start+size-1]
            push!(blocks[level], copy(view_block))
            return
        end

        mid = cld(size, 2)

        # Coordinates of sub-blocks
        i1, i2 = i_start, i_start + mid
        j1, j2 = j_start, j_start + mid

        # Store off-diagonal block as needed
        if triangular_type == :lower
            view_block = @view A[i2:i_start+size-1, j1:j1+mid-1]
            push!(blocks[level], copy(view_block))
        elseif triangular_type == :upper
            view_block = @view A[i1:i1+mid-1, j2:j_start+size-1]
            push!(blocks[level], copy(view_block))
        end

        # Recurse into diagonals
        subdivide(i1, j1, mid, level + 1)
        subdivide(i2, j2, size - mid, level + 1)
    end

    subdivide(1, 1, n, 1)
    return blocks
end




"""
Index mapping function: changed
"""
# function flat_map(i, j, n, l)
#     level = 1
#     block_offset = 0

#     while true
#         mid = cld(n, 2)
#         upper = i <= mid
#         left  = j <= mid

#         if level == l
#             return level + 1, block_offset + 1, i, j
#         end

#         if upper && !left
#             return level, block_offset + 1, i, j - mid
#         elseif !upper && left
#             return level, block_offset + 1, i - mid, j
#         elseif upper && left
#             n = mid
#             level += 1
#             block_offset = 2 * (block_offset)
#         else
#             i, j = i - mid, j - mid
#             n = n - mid
#             level += 1
#             block_offset = 2 * (block_offset) + 1
#         end
#     end
# end

function flat_map(i::Int, j::Int, n::Int, l::Int, triangular_type::Symbol)
    level = 1
    size = n
    diag_path = 0
    block_counts = fill(0, l+1)

    while true
        mid = cld(size, 2)
        upper = i <= mid
        left  = j <= mid

        if level == l + 1
            block_counts[level] += 1
            block_idx = block_counts[level]
            return level, block_idx, i, j
        end

        if triangular_type == :lower && !upper && left
            block_counts[level] += 1
            return level, block_counts[level], i - mid, j
        elseif triangular_type == :upper && upper && !left
            block_counts[level] += 1
            return level, block_counts[level], i, j - mid
        elseif upper && left
            size = mid
            level += 1
            diag_path = 2 * diag_path
        else
            i -= mid
            j -= mid
            size -= mid
            level += 1
            diag_path = 2 * diag_path + 1
        end
    end
end

"""
Efficient indexing for our custom block matrix
"""
# function fast_get(B::TriangularBlockMatrix, i, j)
#     if B.triangular_type == :lower && j > i
#         return zero(eltype(B.blocks[1][1]))
#     elseif B.triangular_type == :upper && i > j
#         return zero(eltype(B.blocks[1][1]))
#     end
#     level, block_idx, local_i, local_j = flat_map(i, j, B.n, B.l)
#     return B.blocks[level][block_idx][local_i, local_j]
# end

function fast_get(B::TriangularBlockMatrix, i, j)
    if B.triangular_type == :lower && j > i
        return zero(eltype(B.blocks[1][1]))
    elseif B.triangular_type == :upper && i > j
        return zero(eltype(B.blocks[1][1]))
    end

    level, block_idx, local_i, local_j = flat_map(i, j, B.n, B.l, B.triangular_type)
    return B.blocks[level][block_idx][local_i, local_j]
end

# Optionally, overload getindex:
Base.getindex(B::TriangularBlockMatrix, i::Int, j::Int) = fast_get(B, i, j)