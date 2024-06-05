export v_mult!

@kernel function v_mult_kernel!(a, b, c)
    i = @index(Global, Linear)
    c[i] = a[i] * b[i]
end

function v_mult!(a, b, c; n_threads = 256)
    if typeof(a) != typeof(b) != typeof(c)
        error("Types of a, b, and c are different!")
    end

    if length(a) != length(b) != length(c)
        error("Lengths of a, b, and c are different!")
    end

    backend = get_backend(a)
    kernel = v_mult_kernel!(backend, n_threads)
    kernel(a, b, c; ndrange = length(a))
end