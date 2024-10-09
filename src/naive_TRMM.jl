export naive_TRMM!

@kernel function naive_TRMM_kernel!(a, b, c)
    i,j = @index(Global, NTuple)

    # a x b = c => c[row_i, col_j] = sum(a[row_i] * b[col_j])


    # only iterate through the lower triangular part

    temp_sum = 0

    for a_col in 1:i
        temp_sum = temp_sum + a[i,a_col] * b[a_col, j]
    end


    c[i,j] = temp_sum
end

function naive_TRMM!(a, b, c; n_threads = 256)

    if size(a)[2] != size(b)[1]
        error("Matrices a and b are incompatible for multiplication!")
    end

    if size(a)[1] != size(c)[1] || size(b)[2] != size(c)[2]
        error("Matrices c dimensions are incompatible to store the product!")
    end

    backend = get_backend(a)
    kernel = naive_TRMM_kernel!(backend, n_threads)
    kernel(a, b, c; ndrange = size(c))
end
