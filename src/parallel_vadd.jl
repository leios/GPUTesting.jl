export par_vadd!

import KernelAbstractions.Extras: @unroll

@kernel function par_vadd_kernel!(a, b, c)

    @uniform gs = @groupsize()
    tile = @localmem eltype(a) gs

    i = @index(Local, Linear)
    I = @index(Global, Linear)

    @inbounds tile[i] = a[I] + b[I]

    @synchronize

    @inbounds c[I] = tile[i]
    
end

function par_vadd!(a, b, c; nthreads = 256)
    if typeof(a) != typeof(b) != typeof(c)
        error("Types of a, b, and c are different!")
    end

    if length(a) != length(b) != length(c)
        error("Lengths of a, b, and c are different!")
    end

    backend = get_backend(a)

    kernel = par_vadd_kernel!(backend, nthreads)
    kernel(a, b, c; ndrange = length(a))
end
