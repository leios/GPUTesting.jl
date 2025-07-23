# export TriangularBlockMatrix, build_block_structure, fast_get


struct TriangularBlockMatrix{T}
    blocks::Vector{Vector{Matrix{T}}}
    n::Int
    l::Int
    triangular_type::Symbol  # :lower or :upper
    # No need for U_approx if l is a direct input
    # Will switch back to U_approx when working with mixed precision
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

    blocks = build_block_structure(A, l; triangular_type)
    return TriangularBlockMatrix{T}(blocks, n, l, triangular_type)
end

"""
build_block_structure: computes and returns the block structure of a matrix
(A is assumed to be square and triangular as per TriangularBlockMatrix constructor)
"""
function build_block_structure(A::AbstractMatrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    blocks = [Matrix{T}[] for _ in 1:l+1]

    # This inner function will populate the `blocks` array.
    # The `block_indices` array will map `diag_path` to the actual `block_idx` (0-indexed)
    # in the `blocks[level]` array. This is critical for `flat_map`.
    # A dictionary would be more robust for sparse `diag_path`s, but for dense trees, an array is fine.
    # We need a way to manage the block index when pushing.
    # Let's collect the blocks and then assign them in order.

    # This revised approach will push the blocks into a flat list for each level
    # and then build the final `blocks` structure with correct 0-indexed block_idx.
    
    # Temp storage to collect blocks in traversal order
    temp_blocks_by_level = [[] for _ in 1:l+1]

    function subdivide_and_collect(i_start::Int, j_start::Int, current_size::Int, level::Int)
        if level == l + 1
            view_block = @view A[i_start:i_start+current_size-1, j_start:j_start+current_size-1]
            push!(temp_blocks_by_level[level], copy(view_block))
            return
        end

        mid = cld(current_size, 2)

        # Coordinates of sub-blocks
        i1, i2 = i_start, i_start + mid
        j1, j2 = j_start, j_start + mid

        # Store off-diagonal block as needed (order matters for flat_map!)
        if triangular_type == :lower
            # A21 block
            view_block = @view A[i2:i_start+current_size-1, j1:j1+mid-1]
            push!(temp_blocks_by_level[level], copy(view_block))
        elseif triangular_type == :upper
            # A12 block
            view_block = @view A[i1:i1+mid-1, j2:j_start+current_size-1]
            push!(temp_blocks_by_level[level], copy(view_block))
        end

        # Recurse into diagonals (always TL then BR)
        subdivide_and_collect(i1, j1, mid, level + 1) # Top-left child
        subdivide_and_collect(i2, j2, current_size - mid, level + 1) # Bottom-right child
    end

    subdivide_and_collect(1, 1, n, 1) # Start recursion

    # Now, populate the `blocks` array, mapping diag_path to block_idx.
    # This requires `flat_map` to determine `diag_path`.
    # The simplest way to map `diag_path` to `block_idx` is if the blocks are pushed
    # in an order that aligns with the `diag_path` directly.
    # The current `subdivide_and_collect` pushes off-diagonal blocks based on a pre-order traversal of the diagonal blocks.
    # So, the `block_idx` for a given `diag_path` would be its position in a sorted list of `diag_path`s at that level.
    # This is getting complicated.

    # Let's revert to a simpler `build_block_structure` that directly assigns based on `diag_path`
    # and requires `blocks[level]` to be sized correctly upfront. This is often better for performance.

    # --- Simplified `build_block_structure` for fixed block_idx (0-indexed `diag_path`) ---
    # Max blocks at level l+1 is 2^l. Max off-diagonal blocks at level k is 2^(k-1).
    max_blocks_level_l_plus_1 = 2^l
    blocks[l+1] = Vector{Matrix{T}}(undef, max_blocks_level_l_plus_1)
    
    for k in 1:l
        num_off_diagonal_blocks = 2^(k-1)
        blocks[k] = Vector{Matrix{T}}(undef, num_off_diagonal_blocks)
    end

    function subdivide_and_assign(i_start::Int, j_start::Int, current_size::Int, level::Int, diag_path::Int)
        if level == l + 1
            # Leaf block (diagonal)
            view_block = @view A[i_start:i_start+current_size-1, j_start:j_start+current_size-1]
            blocks[level][diag_path + 1] = copy(view_block) # diag_path is 0-indexed, so +1 for 1-based Julia array
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
            blocks[level][diag_path + 1] = copy(view_block)
        elseif triangular_type == :upper
            # A12 block
            view_block = @view A[i1:i1+mid-1, j2:j_start+current_size-1]
            blocks[level][diag_path + 1] = copy(view_block)
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