using KernelAbstractions

@kernel function kernel_check(a)
    idx = @index(Global, Linear)
    val = idx

    while val > 1
        #val /= 2
        val = div(val,2)
    end
end

function check(a)
    backend = get_backend(a)
    kernel_check(backend)(a, ndrange = length(a))
end
