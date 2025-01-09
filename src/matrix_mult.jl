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
    @synchronize
end

@kernel function lmem_copy_kernel!(output, @Const(input), 
                                                    ::Val{BANK} = Val(1),) where BANK
    I, J = @index(Global, NTuple)
    i, j = @index(Local, NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N + BANK, M)

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    @inbounds output[I, J] = tile[i, j]
end


function matrix_mult!(a, b, c; n_threads = 256)
    if typeof(a) != typeof(b) != typeof(c)
        error("Types of a, b, and c are different!")
    end

    if size(a)[2] != size(b)[1]
        error("Matrices a and b are incompatible for multiplication!")
    end

    if size(a)[1] != size(c)[1] || size(b)[2] != size(c)[2]
        error("Matrices c dimensions are incompatible to store the product!")
    end

    d = similar(b)
    backend = get_backend(a)
    kernel = matrix_mult_mult_kernel!(backend, n_threads)
    kernel(a, b, d; ndrange = size(d))
    padded_d = (size(d,1)+16, size(d,2)+16)
    lmem_copy_kernel!(backend, (16, 16))(b, d; ndrange = padded_d)
    

end
