export build_block_structure, fast_get

"""
build_block_structure: computes and returns the block structure of a matrix

inputs: A: the square matrix (will be specialized for triangular matrices)
        U_approx: the approximated adaptive precisions for each level

output: A_blocks: format Vector{Vector{Matrix{T}}} where 
                    A_blocks is an Array of the individual arrays for each level
                    A_blocks[i] is the Array of blocks for a particular level i
                    A_blocks[i][k] is the matrix of elements in a block k of level i


"""

function build_block_structure(A, U_approx)
    n = size(A, 1)
    l = size(U_approx, 1) # the number of levels

    @assert size(A, 1) == size(A, 2) "Matrix A must be square."

    blocks = [Matrix[] for _ in 1:l+1]

    function subdivide(A, level)
        if level == l+1
            push!(blocks[level], A)
            return
        end

        n = size(A, 1)
        mid = cld(n, 2)

        top    = 1:mid
        bottom = (mid + 1):n
        left   = 1:mid
        right  = (mid + 1):n


        # format compatible for casting into other precisions
        A11 = A[top, left]
        A12 = A[top, right]
        A21 = A[bottom, left]
        A22 = A[bottom, right]

        # Store only off-diagonal blocks at this level,
        # Will be removing one off diagonal block for triangular matrices
        push!(blocks[level], A12)
        push!(blocks[level], A21)

        # Recurse on diagonals
        subdivide(A11, level + 1)
        subdivide(A22, level + 1)
    end

    subdivide(A, 1)
    return blocks
end




"""
flat_map(i, j, n, l) -> (level, block_index, local_i, local_j)

Maps an index (i, j) in the original n by n matrix to the corresponding
block location in a recursive block structure of depth l.

The block structure is built such that:
- For levels 1 to l, only off-diagonal blocks (A12 and A21) are stored.
- Diagonal blocks (A11 and A22) are recursively subdivided.
- At level l+1, the final diagonal submatrices are stored as full blocks.

Returns:
- level::Int: the level of the block containing the element
- block_index::Int: index into blocks[level]
- local_i::Int, local_j::Int: coordinates inside the block

This function enables efficient and non-recursive access into the hierarchical block storage.
"""
function flat_map(i, j, n, l)
    level = 1
    block_offset = 0  # cumulative block index offset per level

    while true
        mid = cld(n, 2)

        # Determine quadrant
        upper = i <= mid
        left  = j <= mid

        if level == l
            # At level l, store entire submatrix
            # Each diagonal path gives one full block
            block_idx = block_offset + 1
            return level, block_idx, i, j
        end

        if upper && !left
            # A12 (top right) — off-diagonal
            local_i, local_j = i, j - mid
            return level, block_offset + 1, local_i, local_j
        elseif !upper && left
            # A21 (bottom left) — off-diagonal
            local_i, local_j = i - mid, j
            return level, block_offset + 2, local_i, local_j
        elseif upper && left
            # A11 (top-left diagonal) — go deeper
            i, j = i, j
            n = mid
            level += 1
            block_offset = 2 * (block_offset)      # descend diagonal path
        else
            # A22 (bottom-right diagonal) — go deeper
            i, j = i - mid, j - mid
            n = n - mid
            level += 1
            block_offset = 2 * (block_offset) + 1  # descend diagonal path
        end
    end
end




"""
fast_get(blocks, i, j, n, l) -> element

Retrieves the element at position (i, j) in the original n by n matrix,
using the hierarchical block storage structure blocks of depth l.

Arguments:
- blocks: a vector of block arrays as returned by build_block_structure
- i, j: coordinates of the element in the original matrix
- n: size of the original matrix (must be square)
- l: maximum subdivision level used during block structure creation

Returns:
- The scalar element at (i, j) by locating and indexing the appropriate sub-block.

Relies on flat_map for efficient block resolution.
"""
function fast_get(blocks, i, j, n, l)
    # need to use l+1 instead in the function call
    level, block_idx, local_i, local_j = flat_map(i, j, n, l)
    return blocks[level][block_idx][local_i, local_j]
end