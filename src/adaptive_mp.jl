# Function to compute the Frobenius norm of a lower triangular matrix
# Will be extended to other triangular matrix configurations
# Serial, need to parallelize after accuracy confirmed
# Input : L ; Lower Triangular matrix
# Output : float; Frobenius norm of L
function frobenius_norm_LT(L::AbstractMatrix)
    m, n = size(L)
    norm_squared = 0.0

    for i in 1:m
        for j in 1:min(i, n)  # Only consider the lower triangle elements
            norm_squared += abs2(L[i, j])
        end
    end

    return sqrt(norm_squared)
end


# Function to compute the importance of off-diagonal  blocks at level k
# Assume that level 0 is the triangular matrix itself
# importance E_k = (max(Frobenius_norms at level k))/(Frobenius norm of triangular matrix)
# Configure to return all the values and the maximum of the values
# Included the limit to ensure remeaining matrix is still large enough

function diagonal_importance(L::AbstractMatrix, k, limit = 4)
    max_norm = max_frobenius_norm_k(L, k)
    matrix_norm = frobenius_norm_LT(L)
    E_k = max_norm/matrix_norm

    return E_k    
end

# returns the maximum frobenius norm at level k
function max_frobenius_norm_k(L, k, limit = 4)
    f_norms = Float64[]
    # min_size = max(limit, div(size(L,1), k)) To be imposed later
    recursive_frob_norms(L, k, f_norms)
    return max(f_norms)
end


function recursive_frob_norms(sub_matrix, k, f_norms)
    if (k == 0)
        norm = frobenius_norm_LT(sub_matrix)
        push!(f_norms, norm)
    else
        mid = div(size(sub_matrix,1), 2)
        sub_matrixA = @view sub_matrix[1:mid, 1,mid]
        sub_matrixB = @view sub_matrix[mid+1:end, mid+1:end]
        recursive_frob_norms(sub_matrixA, k-1, f_norms)
        recursive_frob_norms(sub_matrixB, k-1, f_norms)
    end
end

# Function to select the adaptive precision at each level
# u_k <= epsilon/(2^(k/2) * E_k)
# E_k : vector of the importance computed for the k levels
# U : vector of the available precision levels expressed as the unit roundoff errors 
# 
function adaptive_precision_k(E_k, k, epsilon = 1e-8)
    # creating the array of roundoff errors
    u16 = eps(Float16)/2
    u32 = eps(Float32)/2
    u64 = eps(Float64)/2
    U = Float64[u16, u32, u64]

    prec = epsilon/(2^(k/2)*E_k)

    u_index = 1

    while ((prec > U[u_index]) && u_index < 3)
        u_index+=1
    end

    return u_index

end



# 1. What is the epsilon?  10^-8 tolerance for the singular matrix
# 2. Understand the derivation of the unit roundoff errors
# 3. Understand proof from Carson(2025) and how the global error remains controlled
# 4. How to select the vector of roundoff errors and what to return as a result
# 5. Is there a function to directly compute the F norm of the LT matrix directly
# 6. Read the different mixed precision


# save as copies of the original 