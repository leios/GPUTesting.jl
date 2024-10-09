export native_TRMM!
include("performant_matrix_mult.jl")


function native_TRMM!(A, B)

    if size(A)[2] != size(B)[1]
        error("Matrices a and b are incompatible for multiplication!")
    end

    
    M = size(A)[1]
    N = size(B)[2]
    m = M/2

    C = similar(A, (m, N))

    perf_mat_mul!(A[1:m, 1:m], B[1:m, 1:N], C)

    B[1:m, 1:N] = C

    
    perf_mat_mul!(A[m+1:M, 1:m], B[m+1:M, 1:N], C)

    B[1:m, 1:N] = C + B[1:m, 1:N]

    K = similar(A, (m, N))

    perf_mat_mul!(A[m+1:M, m+1:M], B[m+1:M, 1:N], K)

    B[m+1:M, 1:N]  = K
    
end
