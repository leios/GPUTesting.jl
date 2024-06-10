export matrix_mult!

@kernel function matrix_mult_mult_kernel!(a, b, c)
    i,j = @index(Global, NTuple)

    # a x b = c => c[row_i, col_j] = sum(a[row_i] * b[col_j])


    #loop for addition, iterations = size of col of a or row of b

    temp_sum = 0

    for a_col in 1:size(a)[2]
        temp_sum = temp_sum + a[i,a_col] * b[a_col, j]
    end


    c[i,j] = temp_sum
end

function matrix_mult!(a, b, c; n_threads = 256)
    if typeof(a) != typeof(b) != typeof(c)
        error("Types of a, b, and c are different!")
    end

    if size(a)[2] != size(b)[1] != size(c)[]
        error("Lengths of a, b, and c are different!")
    end

    backend = get_backend(a)
    kernel = v_add_kernel!(backend, n_threads)
    kernel(a, b, c; ndrange = length(a))
end
