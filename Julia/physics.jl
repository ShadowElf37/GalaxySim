using StaticArrays

#https://math.nyu.edu/~greengar/shortcourse_fmm.pdf

@kwdef mutable struct Star
    x::ComplexF64 = rand(ComplexF64)
    v::ComplexF64 = rand(ComplexF64)
    m::Float64 = 1.0
end

function evolve!(star::Star, dt::Float64=1.0)
    star.x += star.v * dt
end
function push_towards!(star::Star, p::ComplexF64, phi::ComplexF64, dt::Float64=1.0)
    star.v += complex_phi_to_force_magnitude_sq(phi) * (p-star.x) * dt / star.m
end


function ak(stars::AbstractArray{Star}, stars_center::ComplexF64, k::Int)
    -sum(star.m * (star.x-stars_center)^k for star in stars; init = 0.) / k
end
function multipole_phi(stars::AbstractArray{Star}, stars_center::ComplexF64, p::ComplexF64; K_lim::Int64=0)
    # WARNING: p must have |p| > |x| for every x in stars
    local M::Float64 = sum(star.m for star in stars; init = 0.)

    M*log(p-stars_center) + sum(ak(stars, stars_center, k)/(p-stars_center)^k for k in 1:K_lim; init = 0.)
end

function bl(stars::AbstractArray{Star}, stars_center::ComplexF64, l::Int, M::Float64)
    -M*stars_center^l/l + sum(ak(stars, stars_center, k) * (stars_center)^(l-k) * binomial(l-1, k-1) for k in 1:l; init = 0.) / k
end
function multipole_phi_b(stars::AbstractArray{Star}, stars_center::ComplexF64, new_center::ComplexF64, p::ComplexF64; L_lim::Int64=0)
    # WARNING: p must have |p| > |x| for every x in stars
    local M::Float64 = sum(star.m for star in stars; init = 0.)

    M*log(p-new_center) + sum(bl(stars, stars_center, l, M)/(p-new_center)^l for l in 1:L_lim; init = 0.)
end


function complex_phi_to_force_magnitude_sq(phi::ComplexF64)
    return exp(2*real(phi))
end