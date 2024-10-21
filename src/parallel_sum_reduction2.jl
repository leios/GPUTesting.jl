export sumGPU2!

using KernelAbstractions: @atomic

@kernel function sumGPU_kernel2!(input_array, partial_sum)
    gi = @index(Global, Linear)
    i = @index(Local, Linear)

    group_id = @index(Group, Linear)

    #get the groupsize
    @uniform group_size =  @groupsize()[1]

    #allocating shared memory for the local_sums
    tile = @localmem eltype(input_array) group_size
    @uniform input_length = length(input_array)

    @inbounds tile[i] = 0.0


    I = (group_id-1)*group_size+i
    if I <= input_length
        @inbounds tile[i] = input_array[I]
    end 
    @synchronize

    offset = @private Int32 (1,)

    @inbounds begin
        offset[1] = div(group_size, 2)
        while (offset[1] > 1)
            @synchronize 
            if (i < (offset[1]+1))
                tile[i] += tile[i + offset[1]]
            end
            offset[1] = div(offset[1],2)
        end
    end
    

    @inbounds partial_sum[group_id] = tile[1]     
end

function sumGPU2!(a; nthreads = 256)
    b = similar(a, div(length(a) + nthreads, nthreads-1)+5)
    backend = get_backend(a)

    kernel = sumGPU_kernel2!(backend, nthreads)
    kernel(a, b; ndrange = length(a))
    return sum(b)
end
