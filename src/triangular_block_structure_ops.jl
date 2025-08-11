

"""
Block-wise Triangular Matrix Multiplication (TRMM).
Computes C = A * B where A is a TriangularBlockMatrix and B is a standard Matrix.
The result C is a standard Matrix.
"""
function trmm(A_tbm::TriangularBlockMatrix{T}, B_mat::AbstractMatrix{T}) where T
    A_tbm.n != size(B_mat, 1) && error("Dimension mismatch: A.n must equal size(B, 1)")
    
    # Initialize the result matrix C
    C_mat = Matrix{T}(undef, A_tbm.n, size(B_mat, 2))

    # Recursive helper function for block multiplication
    function _trmm_recursive!(
        C_sub::AbstractMatrix{T}, # Subview of C_mat to write results into
        A_tbm_root::TriangularBlockMatrix{T}, # The full A_tbm for block access
        B_sub::AbstractMatrix{T}, # Subview of B_mat
        level::Int,
        diag_path::Int, # diag_path for the current diagonal block of A
        current_size::Int
    )
        # Base case: At the deepest level, perform standard matrix multiplication
        if level == A_tbm_root.l + 1
            A_leaf_block = A_tbm_root.blocks[level][diag_path + 1]
            mul!(C_sub, A_leaf_block, B_sub) # Use mul! for in-place multiplication  ... make sure it is CUBLAS GEMM
            return
        end

        mid = cld(current_size, 2)

        # Split C_sub and B_sub into quadrants
        C11 = @view C_sub[1:mid, 1:size(C_sub, 2)]
        C21 = @view C_sub[mid+1:end, 1:size(C_sub, 2)]

        B11 = @view B_sub[1:mid, 1:size(B_sub, 2)]
        B21 = @view B_sub[mid+1:end, 1:size(B_sub, 2)]

        # Get A's blocks
        # A11 and A22 are handled by recursive calls
        A_off_diag_block = A_tbm_root.blocks[level][diag_path + 1] # This is A21 for lower, A12 for upper

        if A_tbm_root.triangular_type == :lower
            # C = [ A11*B11 ; A21*B11 + A22*B21 ]
            
            # C11 = A11 * B11 (recursive call for top-left diagonal child)
            _trmm_recursive!(C11, A_tbm_root, B11, level + 1, 2 * diag_path, mid)

            # C21 = A21 * B11 + A22 * B21
            # A21 is A_off_diag_block
            # A22 is the bottom-right diagonal child (recursive call)

            # Compute A21 * B11 (temporary allocation)
            temp_prod = A_off_diag_block * B11 
            
            # Add A22 * B21 (recursive call for bottom-right diagonal child)
            # This needs to be accumulated into C21.
            # We can't use mul! directly if B21 is not a sub-block of B_mat.
            # We need to compute A22*B21 and then add it to temp_prod.
            # Or, modify _trmm_recursive! to take an accumulation matrix.
            
            # Simpler: Compute A22 * B21 into a temp, then add.
            temp_prod_A22_B21 = Matrix{T}(undef, size(A_off_diag_block, 1), size(B_sub, 2)) # Size of A22*B21
            _trmm_recursive!(temp_prod_A22_B21, A_tbm_root, B21, level + 1, 2 * diag_path + 1, current_size - mid)
            
            # Accumulate into C21
            C21 .= temp_prod .+ temp_prod_A22_B21

        elseif A_tbm_root.triangular_type == :upper
            # C = [ A11*B11 + A12*B21 ; A22*B21 ]
            
            # C21 = A22 * B21 (recursive call for bottom-right diagonal child)
            _trmm_recursive!(C21, A_tbm_root, B21, level + 1, 2 * diag_path + 1, current_size - mid)

            # C11 = A11 * B11 + A12 * B21
            # A11 is the top-left diagonal child (recursive call)
            # A12 is A_off_diag_block
            # B21 is B_sub[mid+1:end, :]

            # Compute A11 * B11 into a temp
            temp_prod_A11_B11 = Matrix{T}(undef, mid, size(B_sub, 2))
            _trmm_recursive!(temp_prod_A11_B11, A_tbm_root, B11, level + 1, 2 * diag_path, mid)

            # Accumulate A12 * B21
            C11 .= temp_prod_A11_B11 .+ (A_off_diag_block * B21)
        end
    end

    # Start the recursive multiplication
    _trmm_recursive!(C_mat, A_tbm, B_mat, 1, 0, A_tbm.n)

    return C_mat
end


# --- NEW: Block-wise Triangular Matrix Solve (TRMS) ---
"""
Block-wise Triangular Matrix Solve (TRMS).
Computes X such that A * X = B, where A is a TriangularBlockMatrix and B is a standard Matrix.
The result X is a standard Matrix.
"""
function trms(A_tbm::TriangularBlockMatrix{T}, B_mat::AbstractMatrix{T}) where T
    A_tbm.n != size(B_mat, 1) && error("Dimension mismatch: A.n must equal size(B, 1)")
    
    # Initialize the result matrix X
    X_mat = Matrix{T}(undef, A_tbm.n, size(B_mat, 2))

    # Recursive helper function for block solve
    function _trms_recursive!(
        X_sub::AbstractMatrix{T}, # Subview of X_mat to write results into
        A_tbm_root::TriangularBlockMatrix{T}, # The full A_tbm for block access
        B_sub::AbstractMatrix{T}, # Subview of B_mat
        level::Int,
        diag_path::Int, # diag_path for the current diagonal block of A
        current_size::Int
    )
        # Base case: At the deepest level, perform standard matrix solve
        if level == A_tbm_root.l + 1
            A_leaf_block = A_tbm_root.blocks[level][diag_path + 1]
            X_sub .= A_leaf_block \ B_sub # Standard solve for leaf blocks
            return
        end

        mid = cld(current_size, 2)

        # Split X_sub and B_sub into quadrants
        X11 = @view X_sub[1:mid, 1:size(X_sub, 2)]
        X21 = @view X_sub[mid+1:end, 1:size(X_sub, 2)]

        B11 = @view B_sub[1:mid, 1:size(B_sub, 2)]
        B21 = @view B_sub[mid+1:end, 1:size(B_sub, 2)]

        # Get A's blocks
        A_off_diag_block = A_tbm_root.blocks[level][diag_path + 1] # This is A21 for lower, A12 for upper

        if A_tbm_root.triangular_type == :lower
            # A_tbm * X = B
            # [A11 0; A21 A22] * [X11; X21] = [B11; B21]
            # A11*X11 = B11  => X11 = A11 \ B11
            # A21*X11 + A22*X21 = B21 => A22*X21 = B21 - A21*X11 => X21 = A22 \ (B21 - A21*X11)

            # 1. Solve for X11 = A11 \ B11 (recursive call for top-left diagonal child)
            _trms_recursive!(X11, A_tbm_root, B11, level + 1, 2 * diag_path, mid)

            # 2. Compute B21_prime = B21 - A21 * X11
            # A21 is A_off_diag_block
            # X11 is now in X11 (subview of X_mat)
            B21_prime = B21 - A_off_diag_block * X11

            # 3. Solve for X21 = A22 \ B21_prime (recursive call for bottom-right diagonal child)
            _trms_recursive!(X21, A_tbm_root, B21_prime, level + 1, 2 * diag_path + 1, current_size - mid)

        elseif A_tbm_root.triangular_type == :upper
            # A_tbm * X = B
            # [A11 A12; 0 A22] * [X11; X21] = [B11; B21]
            # A22*X21 = B21 => X21 = A22 \ B21
            # A11*X11 + A12*X21 = B11 => A11*X11 = B11 - A12*X21 => X11 = A11 \ (B11 - A12*X21)

            # 1. Solve for X21 = A22 \ B21 (recursive call for bottom-right diagonal child)
            _trms_recursive!(X21, A_tbm_root, B21, level + 1, 2 * diag_path + 1, current_size - mid)

            # 2. Compute B11_prime = B11 - A12 * X21
            # A12 is A_off_diag_block
            # X21 is now in X21 (subview of X_mat)
            B11_prime = B11 - A_off_diag_block * X21

            # 3. Solve for X11 = A11 \ B11_prime (recursive call for top-left diagonal child)
            _trms_recursive!(X11, A_tbm_root, B11_prime, level + 1, 2 * diag_path, mid)
        end
    end

    # Start the recursive solve
    _trms_recursive!(X_mat, A_tbm, B_mat, 1, 0, A_tbm.n)

    return X_mat
end
