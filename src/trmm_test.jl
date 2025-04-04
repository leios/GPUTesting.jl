@testset "Equivalence Test for TRMM: All Cases" begin
    # Matrix sizes to test
    sizes = [16, 32, 128, 256, 2048]

    # Number of columns in B to test
    m_sizes = [1, 8, 64]

    # Tolerance for check
    tolerance = 1e-12



    for n in sizes

        for m in m_sizes

            for side in ['L', 'R'] # side parameter in the trmm function

                for uplo in ['L', 'U'] # uplu parameter in the trmm function

                    for trans in ['N', 'T', 'C'] # whether the parameter A is transposed, also serves as parameter

                        for alpha in [1.0] # Setting the alpha parameter as a constant
                            
                            # Test configuration
                            println("Testing TRMM ; side: $side, uplo: $uplo, trans: $trans, alpha: $alpha, n: $n, m: $m")

                            # Generate the triangular matrix A based on `uplo`
                            if uplo == 'L'
                                # Lower triangular matrix
                                A = Matrix(LowerTriangular(rand(n, n) .+ 1))
                            else
                                # Upper triangular matrix
                                A = Matrix(UpperTriangular(rand(n, n) .+ 1))
                            end

                            # Add a diagonal to ensure the matrix is well-conditioned :confirm if necessary for the TRMM case
                            A += Diagonal(10 * ones(n, n))

                            # Convert A to a CuArray for GPU computation : to be tested on supercomputer
                            A_gpu = CuArray(A)

                            # Generate the B matrix based on the `side`
                            if side == 'L'
                                B = Matrix(rand(n, m) .+ 1)  # B has n rows
                            else
                                B = Matrix(rand(m, n) .+ 1)  # B has n columns
                            end

                            # Create copies of A and B for baseline and comparison
                            Ac = copy(A)
                            Bc = copy(B)
                            B_gpu = CuArray(B)
                            A_gpu_before = copy(A_gpu)

                            # Perform the GPU operation using `unified_rectrxm!`
                            unified_rectrxm!(side, uplo, trans, alpha, 'M', A_gpu, B_gpu)

                            # Perform the baseline operation using BLAS `trmm!`
                            
                            CUBLAS.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
                            

                            # Compute the Frobenius norm difference (relative error)
                            result_diff = norm(Matrix(B_gpu) - Bc) / norm(Bc)

                            # Log the result difference
                            println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                            # Handle NaN results (indicating an error in the computation)
                            if isnan(result_diff)
                                println("GOT NAN..... SKIPPING FOR NOW")
                            end

                            # Check if the relative error exceeds the tolerance
                            if result_diff >= tolerance
                                println("Test failed for matrix size $n x $n, B size: $(size(B)), trans: $trans")
                                println("Relative error: $result_diff")
                            end

                            # Assert that the relative error is within the tolerance
                            @test result_diff < tolerance
                        end
                    end
                end
            end
        end
    end
end