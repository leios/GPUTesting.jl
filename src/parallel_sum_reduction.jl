export sumGPU!

@kernel function sumGPU_kernel!(input_array, partial_sum)
    gi = @index(Global, Linear)
    i = @index(Local, Linear)

    group_id = @index(Group, Linear)

    #get the groupsize
    @uniform group_size =  @groupsize()[1]

    #allocating shared memory for the local_sums
    tile = @localmem eltype(input_array) group_size

    @uniform input_length = length(input_array)

    # private variable for tile output : DEBUGGING
    outval = @private eltype(input_array) 1
    @inbounds outval[1] = -zero(eltype(input_array))

    @uniform end_point = Int(ceil(log(2, group_size)))

   
    #set the tile elements to 0
    @inbounds tile[i] = 0.0

    @synchronize


    I = (group_id-1)*group_size+i
    if I <= input_length
        @inbounds tile[i] = input_array[I]
    else
        tile[i] = 0.0
    end 
    @synchronize


    # The incorrect part
    for k in 1:end_point
        @synchronize

        STRIDE = div(group_size, 2^k)
        I = (group_id-1)*group_size+i
        if (i <= STRIDE)
            @inbounds tile[i] += tile[i+STRIDE]  
        end
    end
    # The incorrect part : saving the first element to the partial sums 

    
    # modifying input array for debugging ###
    I = (group_id-1)*group_size+i
    
    if I <= input_length
        @inbounds input_array[I] = tile[i]
    end
    @synchronize
    

    partial_sum[group_id] = tile[1]
end

function sumGPU!(a; nthreads = 256)
    b = zeros(div(length(a) + nthreads, nthreads-1))
    backend = get_backend(a)

    kernel = sumGPU_kernel!(backend, nthreads)
    kernel(a, b; ndrange = length(a))
    return b
end