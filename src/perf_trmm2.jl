include("gemm_add.jl")


# implementing the base kernels, dependent on the occupied side of A,
# in place, store result in B





# the recursive trmm functions
# The first letter is the triangle referencing, Upper/ Lower
# The second letter is the position of the A argument

function recTRMM_LL(alpha, Afull, Bfull, start_index, end_index,  threshold)
    backend = get_backend(A)

    
    size_A = (end_index - start_index + 1)
    

    if size_A <= threshold
        #Lower_trmm!(backend, )
        

    else
        # The split is the final index of A11
        split = div((start_index + end_index), 2) 
        
        # 1. operate recursively on B1 : B1 = alpha * A11 * B1
        recTRMM_LL(alpha, Afull, Bfull, split+1, end_index, threshold)
        GEMM_ADD!(alpha, Afull)
        

    end
end


# TO DO: Finish implementation of UL
function rec_TRMM_UL(alpha, Afull, Bfull, start_index, end_index, threshold)
    
end


# TO DO: implement recTRMM_LL and rec_TRMM_UL





# the main trmm call.

# CHARACTER ARGUMENTS
# Multiplication Order
# side	Meaning
# 'L'	The argument goes on the left side of a matrix-matrix operation.
# 'R'	The argument goes on the right side of a matrix-matrix operation.

# Triangle Referencing
# uplo/ul	Meaning
# 'U'	    Only the upper triangle of the matrix will be used.
# 'L'	    Only the lower triangle of the matrix will be used.

# Transposition Operation
# trans/tX	Meaning
# 'N'	    The input matrix X is not transposed or conjugated.
# 'T'	    The input matrix X will be transposed.
# 'C'	    The input matrix X will be conjugated and transposed.

# Unit Diagonal
# diag/dX	Meaning
# 'N'	The diagonal values of the matrix X will be read.
# 'U'	The diagonal of the matrix X is assumed to be all ones.

# TRMM/ TRSM ?
# 'S'       Solve
# 'M'       Multiply



# Update B as alpha*A*B
# Return the updated B
function perf_trmm(side, ul, tA, dA, alpha, A, B)
    # assume dA = 'N'
    # call the appropriate TRMM recursive function
    if side == 'L' && ul == 'L' && tA == 'N'
    
    elseif side == 'L' && ul == 'U' && tA == 'N'
    
    else
        error("Unsupported combination of parameters")
    end


end









# Formulation
# Assumptions: A is lower triangular, solving B = AB; A is nxn, B is nxm;

# Recursion:
# We are performing a recursion with a threshold in which we use the base case TRMM.

# If n is less than or equal to the threshold:
#     Call the base case kernel function to carry out the TRMM in place.

# Split the matrix A into 4 equal-sized submatrices, but focus on the 3 with non-zero elements:
#     A11: Top left (lower triangular)
#     A22: Bottom right (lower triangular)
#     A21: Bottom left (full matrix)
#     A12: Top right (empty) *not used in computation

# Split matrix B into two halves:
#     B1: Top half
#     B2: Bottom half

# Call recursion on the top left submatrix (A11) with size n/2 x n/2: recTRMM(A11, B1) : B1 = A11*B1

# Perform GEMM and addition: Update TOP half of B: B1 = A21*B2 + B1

# Call recursion on the bottom right submatrix (A22) with size n/2 x n/2: recTRMM(A22, B2)

# Base Case of the Triangular Solve (TRSM): TO DO

# For each row in B: Get the diagonal element from A for that row, 
# then update the corresponding entry in B by dividing it by the diagonal element.

# Then for each row below the current row, update B by subtracting contributions from rows above it.

# Store the updated value back into B.