#using GLMakie
using StaticArrays
using Random
include("physics.jl")

z_to_index(z::ComplexF64, mesh_size::Int) = CartesianIndex(Int.(floor.((real(z), imag(z)).*mesh_size.+1)))
index_to_z(index::Union{AbstractArray{Int}, Tuple{Int, Int}}, mesh_size::Int) = reinterpret(ComplexF64, (index.-0.5)./mesh_size)[1]
index_to_z(i::Int, j::Int, mesh_size::Int) = index_to_z((i,j), mesh_size)

im2p(p::ComplexF64) = Point2f(real(p), imag(p))
s2p(star::Star) = im2p(star.x)


const N = 100  # number of particles
const ϵ = 0.001  # desired accuracy
const G = 1.  # gravitational const

const p = Int(ceil(-log(2.12, ϵ)))  # number of multipole terms to include
const m = Int(ceil(log2(N)))  # number of mesh levels

println(p)
println(m)

sizes = 2 .^ (m:-1:m-1)
meshes = [zeros(ComplexF64, (n,n)) for n in sizes]
#g_meshes = [zeros(ComplexF64, (n,n)) for n in m:-1:1]

stars = [Star() for i in 1:N]


# START FILL ALL MESH
for (meshidx, msize) in enumerate(sizes)
    in_box = reshape([BitVector(undef, N) for _ in 1:msize, _ in 1:msize], (msize, msize))
    for (i, star) in enumerate(stars)
        in_box[z_to_index(star.x, msize)][i] = 1
    end

    for i in 1:msize
        for j in 1:msize
            for x in 1:msize
                for y in 1:msize
                    if x == i && y == j
                        continue
                    end
                    p_current = index_to_z((i,j), msize)
                    p_there = index_to_z((x,y), msize)
                    meshes[meshidx][x,y] += multipole_phi(stars[in_box[i,j]], p_current, p_there; K_lim=p)#(p_current-p_there) * complex_phi_to_force_magnitude_sq(multipole_phi(stars[in_box[i,j]], p_current, p_there; K_lim=p))
                    #meshes[1][x,y] += (p_current-p_there) * complex_phi_to_force_magnitude_sq(multipole_phi(stars[in_box[i,j]], p_current, p_there; K_lim=p))

                    #finest_mag[x,y] += abs(meshes[1][x,y])
                end
            end
        end
    end
end
# END FILL FINEST MESH
# START FILL NEXT MESH
    #=function fill_coarser_mesh!(finer::AbstractArray{ComplexF64}, coarser::AbstractArray{ComplexF64})
        fine_size = size(finer, 1)
        coarse_size = size(coarser, 1)
        for ci in 1:coarse_size
            for cj in 1:coarse_size
                origin::ComplexF64 = index_to_z(ci, cj, coarse_size)
                finer_indices = [(ci*2, cj*2), (ci*2-1, cj*2), (ci*2, cj*2-1), (ci*2-1, cj*2-1)]
                z0::Array{ComplexF64} = [index_to_z(idx, finer_size) for idx in finer_indices]
                for z in z0
                    coarser[ci, cj] += multipole_phi_b()
                end
            end
        end
    end=#



println((meshes[1][1:5,1:5]))
#println((finest_mag[1:5,1:5]))


using GLMakie
#using Iterators
fig = Figure(size=(400, 600))

ax = Axis(fig[1,1]; title="Finest Mesh", backgroundcolor = "white")
#arrows!(ax, (Iterators.product(LinRange(0, 1, finest_size)), LinRange(0, 1, finest_size)), im2p.(meshes[1]); normalize=true)
h = heatmap!(ax, 0:1/m:1, 0:1/m:1, real.(meshes[1]); colormap = :viridis)
scatter!(ax, map(s2p, stars); color="#fff")
Colorbar(fig[1,2]; label="Gravitational Potential", limits=Float64.(extrema(real.(meshes[1]))))

ax = Axis(fig[2,1]; title="Finest Mesh", backgroundcolor = "white")
#arrows!(ax, (Iterators.product(LinRange(0, 1, finest_size)), LinRange(0, 1, finest_size)), im2p.(meshes[1]); normalize=true)
h = heatmap!(ax, 0:2/m:1, 0:2/m:1, real.(meshes[2]); colormap = :viridis)
scatter!(ax, map(s2p, stars); color="#fff")
Colorbar(fig[2,2]; label="Gravitational Potential", limits=Float64.(extrema(real.(meshes[2]))))

save("test.png",fig)

#println(complex_phi_to_force_magnitude_sq(multipole_phi(stars[1:2], 2.0 + 0.0im)))