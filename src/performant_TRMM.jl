export performant_TRMM!

@kernel function performant_TRMM_kernel(A, B )
    




end



function performant_TRMM!(A, B, LIMIT ; n_threads = (16,16) )

    #resize
     
    k  = 2^(Int(ceil(log(2, size(A)[2]))))
    A_pad = zeros(k, k)
    B_pad = zeros(k ,k)

    A_pad[1 : size(A)[2] , 1:size(A)[2]] .= A
    B_pad[1 : size(A)[2] , 1:size(A)[2]] .= B
    size_a = div(k, 2)

    

    #recursive function
    function recursive_TRMM!(A_2, B_2, size_a)

        # check the size of the matrix and perform recursive splitting
        # base case
        
        
        if size(A_2)[2] <= LIMIT
        # call a base case multiplier: start with naive_TRMM in place
        #step 1
            B_2[size_a + 1: end , 1:size_a]        .= A_2[size_a + 1: end ,size_a+1: end] * B_2[size_a + 1: end ,1:size_a]
            B_2[size_a + 1: end , size_a + 1: end] .= A_2[size_a + 1: end ,size_a+1: end] * B_2[size_a + 1: end ,size_a + 1: end]

        #step 2
            B_2[size_a + 1: end ,1:size_a]        .=  B_2[size_a + 1: end ,1:size_a] + (A_2[size_a + 1: end , 1: size_a] * B_2[1:size_a ,1:size_a])
            B_2[size_a + 1: end ,size_a + 1: end] .=   B_2[size_a + 1: end ,size_a + 1: end] + (A_2[size_a + 1: end , 1: size_a] * B_2[1:size_a ,size_a + 1: end])
        
        #step 3
            B_2[1 : size_a, 1: size_a] .= A_2[ 1: size_a , 1: size_a  ] * B_2[1: size_a , 1: size_a ]
            B_2[1 : size_a , size_a + 1: end] .= A_2[ 1: size_a , 1: size_a  ] * B_2[1: size_a , size_a + 1: end]



        # perform recursive splitting
        else
        #step 1
            B_2[size_a + 1: end , 1:size_a]  .= recursive_TRMM!( A_2[size_a+1: end, size_a+1:end], B_2[size_a+1: end, 1:size_a], div(size_a , 2))
            B_2[size_a + 1: end , size_a + 1: end] .= recursive_TRMM!( A_2[size_a+1: end, size_a+1:end],  B_2[size_a+1: end, size_a+1: end], div(size_a , 2))

        #step 2 GEMM

            B_2[size_a+1: end, 1:size_a] .= B_2[size_a+1: end, 1:size_a] + A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, 1:size_a]
            B_2[size_a+1: end, size_a+1: end] .= B_2[size_a+1: end, size_a+1: end] + A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, size_a + 1: end]

        #step 3
            B_2[1 : size_a, 1: size_a] .= recursive_TRMM!(A_2[1: size_a, 1: size_a],  B_2[1: size_a, 1: size_a], div(size_a , 2))
            B_2[1 : size_a , size_a + 1: end] .= recursive_TRMM!( A_2[1: size_a, 1: size_a],  B_2[1: size_a, size_a+1: end], div(size_a , 2))
        end

        return B_2


    end

    B_pad .= recursive_TRMM!(A_pad, B_pad, div(k,2))

    B = @view(B_pad[1: size(A)[2] , 1 : size(A)[2]])

    return B_pad

end