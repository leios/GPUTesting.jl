using Metal
function v_add(a,b,c)
    i = thread_position_in_grid_1d()
    c[i] = a[i] + b[i]
    return
end
a = MtlArray([2,3,4,5])
b = MtlArray([6,7,8,9])
c = similar(a)
@metal threads = 2 groups = 2 v_add(a,b,c)
display(c)
