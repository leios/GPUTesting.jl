export sumGPU!

using KernelAbstractions: @atomic

@kernel function sumGPU_kernel!(input_array, partial_sum)
    gi = @index(Global, Linear)
    i = @index(Local, Linear)

    group_id = @index(Group, Linear)

    #get the groupsize
    @uniform group_size =  @groupsize()[1]

    #allocating shared memory for the local_sums
    tile = @localmem eltype(input_array) group_size

    @uniform input_length = length(input_array)

    @uniform end_point = ceil((log(2, group_size)))

   


    I = (group_id-1)*group_size+i
    if I <= input_length
        @inbounds tile[i] = input_array[I]
    else
        tile[i] = 0.0
    end 
    @synchronize


    # The incorrect part
    for k in 1:end_point
        

        STRIDE = Int32(div(group_size, 2^k))
        I = (group_id-1)*group_size+i
        if (i <= STRIDE)
            @inbounds tile[i] += tile[i+STRIDE]  
        end
        @synchronize
    end
    # The incorrect part : saving the first element to the partial sums 

    if (i == 1)
        @atomic partial_sum[group_id] += tile[i]     
    end
    @synchronize
end

function sumGPU!(a; nthreads = 256)
    b = similar(a,div(length(a) + nthreads, nthreads-1))
    backend = get_backend(a)

    kernel = sumGPU_kernel!(backend, nthreads)
    kernel(a, b; ndrange = length(a))
    return sum(b)
end
