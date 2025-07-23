export TriangularBlockMatrix, build_block_structure, fast_get

using LinearAlgebra

# --- 1. TriangularBlockMatrix Structure ---
struct TriangularBlockMatrix{T}
    #: Now stores AbstractMatrix{T} to allow for views (SubArray)
    blocks::Vector{Vector{AbstractMatrix{T}}}
    n::Int
    l::Int
    triangular_type::Symbol  # :lower or :upper
    # No need for U_approx if l is a direct input
end


"""
Constructs a TriangularBlockMatrix from a square matrix 'A'
and a given number of levels 'l'.
"""
function TriangularBlockMatrix(A::Matrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    if size(A, 1) != size(A, 2)
        error("Input matrix A must be square.")
    end
    
    # Check if A is actually triangular of the specified type
    if triangular_type == :lower && !istril(A)
        error("Input matrix A must be lower triangular for triangular_type = :lower.")
    elseif triangular_type == :upper && !istriu(A)
        error("Input matrix A must be upper triangular for triangular_type = :upper.")
    end

    # Call the build_block_structure that uses the diag_path based indexing and stores views
    blocks = build_block_structure(A, l; triangular_type)
    return TriangularBlockMatrix{T}(blocks, n, l, triangular_type)
end

"""
build_block_structure: computes and returns the block structure of a matrix
(A is assumed to be square and triangular as per TriangularBlockMatrix constructor)
Crucially, this version stores VIEWS (`SubArray`) instead of COPIES to reduce allocations.
"""
function build_block_structure(A::AbstractMatrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    
    # Vector{AbstractMatrix{T}} to store SubArray views
    blocks = Vector{Vector{AbstractMatrix{T}}}(undef, l + 1) 

    # --- Simplified `build_block_structure` for fixed block_idx (0-indexed `diag_path`) ---
    # Max blocks at level l+1 is 2^l. Max off-diagonal blocks at level k is 2^(k-1).
    max_blocks_level_l_plus_1 = 2^l
    blocks[l+1] = Vector{AbstractMatrix{T}}(undef, max_blocks_level_l_plus_1) #AbstractMatrix{T}
    
    for k in 1:l
        num_off_diagonal_blocks = 2^(k-1)
        blocks[k] = Vector{AbstractMatrix{T}}(undef, num_off_diagonal_blocks) #AbstractMatrix{T}
    end

    function subdivide_and_assign(i_start::Int, j_start::Int, current_size::Int, level::Int, diag_path::Int)
        if level == l + 1
            # Leaf block (diagonal)
            view_block = @view A[i_start:i_start+current_size-1, j_start:j_start+current_size-1]
            blocks[level][diag_path + 1] = view_block #Removed copy()
            return
        end

        mid = cld(current_size, 2)

        # Coordinates of sub-blocks
        i1, i2 = i_start, i_start + mid
        j1, j2 = j_start, j_start + mid

        # Assign off-diagonal block using diag_path as index
        if triangular_type == :lower
            # A21 block
            view_block = @view A[i2:i_start+current_size-1, j1:j1+mid-1]
            blocks[level][diag_path + 1] = view_block # Removed copy()
        elseif triangular_type == :upper
            # A12 block
            view_block = @view A[i1:i1+mid-1, j2:j_start+current_size-1]
            blocks[level][diag_path + 1] = view_block # Removed copy()
        end

        # Recurse into diagonals (TL child takes 2*diag_path, BR child takes 2*diag_path + 1)
        subdivide_and_assign(i1, j1, mid, level + 1, 2 * diag_path) # Top-left child
        subdivide_and_assign(i2, j2, current_size - mid, level + 1, 2 * diag_path + 1) # Bottom-right child
    end

    subdivide_and_assign(1, 1, n, 1, 0) # Start recursion with diag_path = 0 for the root
    return blocks
end

# And now, the `flat_map` function is much simpler and pure:
function flat_map(i::Int, j::Int, n::Int, l::Int, triangular_type::Symbol)
    level = 1
    current_size = n
    diag_path = 0 # 0-indexed path to the diagonal ancestor block

    while true
        mid = cld(current_size, 2)
        upper = i <= current_size && j <= current_size && i <= mid
        left  = i <= current_size && j <= current_size && j <= mid

        # Check for hitting an off-diagonal block at the current level
        if triangular_type == :lower && !upper && left # A21 region
            return level, diag_path + 1, i - mid, j # block_idx is 0-indexed diag_path + 1
        elseif triangular_type == :upper && upper && !left # A12 region
            return level, diag_path + 1, i, j - mid # block_idx is 0-indexed diag_path + 1
        end
        
        # If not an off-diagonal block, it must be a diagonal block, recurse
        if upper && left # Top-left diagonal child
            current_size = mid
            level += 1
            diag_path = 2 * diag_path
        elseif !upper && !left # Bottom-right diagonal child
            i -= mid
            j -= mid
            current_size -= mid
            level += 1
            diag_path = 2 * diag_path + 1
        else
            # This case implies an error in logic or a non-triangular query
            # For a strictly triangular matrix, (i,j) must fall into an expected block.
            error("Invalid (i,j) for triangular decomposition: ($i, $j) in block of size $current_size at level $level for $triangular_type matrix.")
        end

        # At the leaf level (l+1), we return the diagonal block
        if level == l + 1
            return level, diag_path + 1, i, j
        end
    end
end

# --- fast_get Function and Base.getindex Overload ---
"""
Efficient indexing for block matrix.
Checks for zero regions first, then uses flat_map for non-zero elements.
"""
function fast_get(B::TriangularBlockMatrix, i::Int, j::Int)
    # Handle zero regions based on triangular type
    if B.triangular_type == :lower && j > i
        return zero(eltype(B.blocks[1][1]))
    elseif B.triangular_type == :upper && i > j
        return zero(eltype(B.blocks[1][1]))
    end

    # If not in a zero region, find the block and local coordinates
    level, block_idx, local_i, local_j = flat_map(i, j, B.n, B.l, B.triangular_type)
    
    # Access the specific block and its element
    return B.blocks[level][block_idx][local_i, local_j]
end

# Optionally, overload getindex for convenient syntax
Base.getindex(B::TriangularBlockMatrix, i::Int, j::Int) = fast_get(B, i, j)
