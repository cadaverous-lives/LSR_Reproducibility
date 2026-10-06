using ForwardDiff, StaticArrays, LinearAlgebra, NonlinearSolve

function mat2svectors(x::AbstractMatrix{T}, ::Val{N}) where {T,N}
    size(x,1) == N || error("sizes mismatch")
    isbitstype(T) || error("use for bitstypes only")
    reinterpret(SVector{N,T}, vec(x))
end

function mat2svectorscopy(x::AbstractMatrix{T}, ::Val{N}) where {T,N}
    copy(mat2svectors(x, Val{N}()))
end

function svectors2mat(x::AbstractVector{SVector{N, T}}) where {N, T}
    isbitstype(T) || error("use for bitstypes only")
    reshape(reinterpret(T, x), (N, length(x)))
end

function svectors2matcopy(x::AbstractVector{SVector{N, T}}) where {N, T}
    copy(svectors2mat(x))
end

# my fix for allocations when computing jacobians along with the primal with forwarddiff on staticarrays
ForwardDiffStaticArraysExt = Base.get_extension(ForwardDiff, :ForwardDiffStaticArraysExt)
import .ForwardDiffStaticArraysExt: extract_jacobian, static_dual_eval

function my_with_jacobian(f::F, x::SVector{N, T}) where {F<:Function, N, T}
    Tg = ForwardDiff.Tag{F, T}
    ydual = static_dual_eval(Tg, f, x)
    y = similar(x)
    ForwardDiff.extract_value!(Tg, nothing, y, ydual)
    return (SVector(y), extract_jacobian(Tg, ydual, x))
end

# my implementation of simple newton's method that doesn't allocate for static arrays
function my_newton(f::F, x0::SVector{N, T}, ps; abstol=1e-10, reltol=0., maxiters=100) where {F, N, T}
    # catch init termination
    # r, J = my_with_jacobian(p->f(p, ps), x0)
    r = f(x0, ps)
    r0 = norm(r)
    # r0 <= abstol && return x0
       
    x = MVector(x0)
    fclos = p->f(p, ps)
    J = ForwardDiff.vector_mode_jacobian(fclos, x0)

    for _ = 1:maxiters
        x .-= J \ r

        norm(r) <= abstol + reltol*r0 && return SVector(x)

        r, J = my_with_jacobian(p->f(p, ps), SVector(x))
    end

    # @warn "Max iters reached"
    return SVector(x)
end

OptionalFunction = Union{Function, Nothing}

# struct ImplicitTimeStepper{P <: Function, R <: Function, RJ0 <: Function, RJ1 <: Function, PJ0 <: OptionalFunction, PJ1 <: OptionalFunction, RT <: Function} <: AbstractTimeDiscretization
#     name::String
#     Φ::P            # approximate implementation of u_i+1 ≈ Φ(u_i+1, u_i, τ_i, h),
#                     # where u_i may be used as an initial guess for nonlinear solver,
#                     # st r_i(u_i+1, u_i, h) = τ_i
#     r!::R           # residual function that implicitly defines the time discretization, r_i = u_i+1 - Φ(u_i+1, u_i, h)
#     residual_J0::RJ0 # Jacobian of r wrt u_i
#     residual_J1::RJ1 # Jacobian of r wrt u_i+1
#     residual_P0::PJ0 # preconditioner for Jacobian of r wrt u_i
#     residual_P1::PJ1 # preconditioner for Jacobian of r wrt u_i+1
#     residual_dt!::RT # derivative of r wrt h
# end

function KKT_gauss_newton(W::AbstractMatrix, res::Function, p0::SVector{Np, T}, ps; pdamp=zero(p0), abstol=1e-10, reltol=0., maxiters=100) where {Np, T}
    r, A = res(p0, ps)

    r0 = norm(r)
    # r0 < abstol && return p0

    nr = length(r)
    p = similar(p0)
    p .= p0

    for k ∈ 1:maxiters
        B = [[W.^2 A']; [A zero(SMatrix{nr, nr, T})]]
        y = [W.^2 * pdamp; r]

        x = B \ y
        δp = x[eachindex(p)]

        p .-= δp

        norm(r) < abstol + reltol*r0 && break

        r, A = res(SVector(p), ps)
    end

    return SVector(p)
end

function KKT_gauss_newton(W::AbstractVector, res::Function, p0::AbstractVector, ps; pdamp=zero(p0), abstol=1e-10, reltol=0., maxiters=100)
    r, A = res(p0, ps)
    r0 = norm(r)
    # r0 < abstol && return p0

    np = length(p0)
    nr = length(r)

    p = similar(p0)
    p .= p0

    B = zeros(np+nr, np+nr)
    y = zeros(np+nr)
    x = zeros(np+nr)

    @views B[diagind(B)[1:np]] .= W.^2
    @views y[1:np] .= W.^2 .* pdamp

    for k ∈ 1:maxiters
        @views y[np+1:end] .= r
        @views B[np+1:end, 1:np] .= A
        @views B[1:np, np+1:end] .= A'

        x .= B \ y
        p .-= x[eachindex(p)]

        norm(r) < abstol + reltol*r0 && break

        r, A = res(p, ps)
    end

    return p
end

# here, p is descent direction, x is current solution guess
function recursive_linesearch_bisect(x, δx, params, res, l0, l1; γ=1., γ_min=.01)
    xh = x - γ/2*δx
    r, F = res(xh, params)
    lh = 1/2*norm(r)^2
    if lh < l0 || γ < γ_min
        return xh, r, F
    else
        return recursive_linesearch_bisect(x, δx, params, res, l0, lh; γ=γ/2)
    end
end

function recursive_linesearch_quadratic(x, δx, params, res, l0, l1; γ=1.)
    xh = x - γ/2*δx
    r, F = res(xh, params)
    lh = 1/2*norm(r)^2
    if lh < l0
        return xh, r, F
    else
        # still no decrease, minimize quadratic fit of h(α) = f(x + α δx)
        c2 = 2*(lh - l0)
        c3 = 2*(l1 - 2lh + l0)
        # @show [l0, lh, l1]
        α = γ*(c3 - 2c2)/(4*c3)
        # @show α
        if c3 > 0.
            xq = x - α*δx
            r, F = res(xq, params)
            return xq, r, F
        else
            return recursive_linesearch_quadratic(x, δx, params, res, l0, lh; γ=γ/2)
        end
    end
end

# my implementation of Gauss-Newton with QR decomposition (requires no autograd!)
# function my_gauss_newton(res::Function, x0::SVector{N, T}, ps; abstol=1e-10, line_search=recursive_linesearch_quadratic, maxiters=100, damping=1.) where {N, T}
function my_gauss_newton(res::Function, x0::SVector{N, T}, ps; abstol=1e-10, maxiters=100, γ=1., verbose=false, linesearch=nothing, weight=nothing) where {N, T}
    x = MVector(x0)
    r, J = res(SVector(x), ps)
    l0 = 1/2*norm(r)^2
    l1 = Inf
    verbose && @show l0

    for i = 1:maxiters
        if !isnothing(weight)
            r = weight*r
            J = weight*J
        end
        Q, R = qr(J)
        δx = R \ Q'*r

        # normal equations are faster for really small systems...
        # δx = (J'*J) \ (J'*r)
        x1 = x - γ*δx

        r, J = res(SVector(x1), ps)
        l1 = 1/2*norm(r)^2

        # check objective decreased
        if !isnothing(linesearch) && l1 > l0
            x2, r, J = linesearch(SVector(x), δx, ps, res, l0, l1; γ=γ)
            x .= x2
        else
            x .= x1
        end

        l1 = 1/2*norm(r)^2
        verbose && @show l1
        if i > 1 && norm(l1 - l0) < abstol
        # if norm(δx) < abstol
            if verbose
                println("Converged in $i")
                break
            end
        end
        l0 = l1
    end

    return SVector(x)
end

# constrained Gauss-Newton
function my_gauss_newton(res::Function, cons::Tuple, x0::SVector{N, T}, ps; abstol=1e-10, maxiters=1, γ=1., verbose=false, linesearch=nothing, weight=nothing) where {N, T}
    x = MVector(x0)
    r, J = res(SVector(x), ps)
    l0 = 1/2*norm(r)^2
    l1 = Inf
    verbose && @show l0

    for i = 1:maxiters
        if !isnothing(weight)
            r = weight*r
            J = weight*J
        end
        # QR decomposition
        Q, R = qr(J)
        δx = -R \ Q'*r

        # normal equations are faster for really small systems...
        # G = J'*J
        # δx = -(G \ (J'*r))

        # satisfy the constraint
        C, c = cons

        # δx = δx - R \ ((C'*δx - c) / (C'*C) * C)
        δx = δx + 0.5 * ((C'*δx - c) / (C'*C) * C)

        # C_proj = R' \ C
        # C_proj = R \ C
        # δx = δx - ((C'*δx - c) / (C'*C_proj)) * C_proj
        # δx = δx - ((C'*δx - c) / (C'*C)) * C_proj
        # δx = δx - ((C'*δx - c) / (C'*C)) * C
        # δx = δx - R \ ((C'*δx - c) / (C_proj'*C_proj) * C_proj)

        # C_proj = G \ C
        # δx = δx - ((C'*δx - c) / (C' * C_proj)) * C_proj

        x .+= γ*δx

        r, J = res(SVector(x), ps)
        l1 = 1/2*norm(r)^2

        verbose && @show l1
        if i > 1 && abs(l1 - l0) < abstol
        # if norm(δx) < abstol
            if verbose
                println("Converged in $i")
                break
            end
        end
        l0 = l1
    end

    return SVector(x)
end


abstract type AbstractDiffEQ end

struct OdeSystem{N, F <: Function, FJ <: Function, FH <: OptionalFunction} <: AbstractDiffEQ
    name::String
    f::F
    J::FJ
    Hv::FH
    OdeSystem{N}(name, f::F, J::FJ, Hv::FH) where {N, F, FJ, FH} = new{N, F, FJ, FH}(name, f, J, Hv)
end

OdeSystem{N}(name::String, f, J) where N = OdeSystem{N}(name, f, J,  nothing)
Base.length(::OdeSystem{N}) where N = N
Base.size(::OdeSystem{N}) where N = (N,)
Base.show(io::IO, sys::OdeSystem{N}) where N = print(io, "OdeSystem{$N}, named \"$(sys.name)\"")

abstract type AbstractTimeDiscretization end

struct TimeStepper{P <: Function, PJ <: Function, PT <: Function, PH <: OptionalFunction, PG <: OptionalFunction } <: AbstractTimeDiscretization
    name::String
    Φ::P         # explicit time step u_i = Φ(u_i, u_i-1, h) (u_i is ignored)
    Φ_u::PJ
    Φ_t::PT
    Φ_Hv::PH
    Φ_withgrads::PG
end

# dalquist system
function dalquist(λ)
    f_λ(u) = λ*u
    J_λ(u) = λ
    Hvλ(v, u) = 0

    OdeSystem{1}("Dalquist (λ = $λ)", f_λ, J_λ, Hvλ)
end

# Lorenz system
const σ = 10
const ρ = 28
const β = 8/3
const lorenz_ps = @SVector[σ, ρ, β]

function f_lorenz(u::AbstractVector{T}; p=lorenz_ps)::SVector{3, T} where T
    x, y, z = u
    σ, ρ, β = T.(p)
    SVector{3, T}(
        σ*(y - x),
        x*(ρ - z) - y,
        x*y - β*z
    )
end

function J_lorenz(u::AbstractVector{T}; p=lorenz_ps)::SMatrix{3, 3, T} where T
    (x, y, z) = u
    (σ, ρ, β) = T.(p)
    @SMatrix T[
        -σ   σ  0
        ρ-z -1 -x
        y    x -β
    ]
end

# adjoint hessian vector product (output is a matrix)
# This is ∂^2f^T/(∂u∂u^T) * v
function Hv_lorenz(v::AbstractVector{T}, u; p=lorenz_ps)::SMatrix{3, 3, T} where T
    @SMatrix T[
        0    v[3] -v[2]
        v[3] 0     0
       -v[2] 0     0
    ]
end

const lorenz = OdeSystem{3}("Lorenz", f_lorenz, J_lorenz, Hv_lorenz)

# diffusionless-Lorenz

function f_lorenz_dfl(u::AbstractVector{T})::SVector{3, T} where T
    x, y, z = u
    SVector{3, T}(
        y - x,
        -x*z,
        x*y - 1.
    )
end

function J_lorenz_dfl(u::AbstractVector{T})::SMatrix{3, 3, T} where T
    (x, y, z) = u
    @SMatrix T[
        -1  1  0
        -z  0 -x
         y  x  0
    ]
end

const lorenz_dfl = OdeSystem{3}("Diffusionless Lorenz", f_lorenz_dfl, J_lorenz_dfl)

# coupled lorenz system
const lorenz_time_scale = 10
const lorenz_fine_coupl = 0.5
const lorenz_crse_coupl = 1.0

function f_lorenz_coupled(u::AbstractVector{T}; p=[σ, ρ, β, lorenz_time_scale, lorenz_fine_coupl, lorenz_crse_coupl]) where T
    x1, y1, z1, x2, y2, z2 = u
    σ, ρ, β, s, c1, c2 = p
    SVector{6, T}(
        σ*(y1 - x1) + c2*x2,
        x1*(ρ - z1) - y1 + c2*y2,
        x1*y1 - β*z1 + c2*(z2 - ρ),
        s*σ*(y2 - x2),
        s*(x2*(ρ - z2 + c1*(z1 - ρ)) - y2),
        s*(x2*y2 - β*z2)
    )
end

function J_lorenz_coupled(u::AbstractVector{T}; p=[σ, ρ, β, lorenz_time_scale, lorenz_fine_coupl, lorenz_crse_coupl]) where T
    u1 = SVector{3, T}(u[SVector(1, 2, 3)])
    u2 = SVector{3, T}(u[SVector(4, 5, 6)])
    σ, ρ, β, s, c1, c2 = p
    J = zero(MMatrix{6, 6, T})
    J[1:3, 1:3] .= J_lorenz(u1)
    J[1:3, 4:6] .= c2*SDiagonal{3, T}(I)
    # bottom left corner is a little funky
    J[5, 3] = s*c1*u2[1]
    J[4:6, 4:6] = s*J_lorenz(u2; p=[σ, ρ+c1*(u1[3]-ρ), β])
    return SMatrix(J)
end

# # const lorenz_coupled = OdeSystem{6}("Two-time Lorenz", f_lorenz_coupled, J_lorenz_coupled)
lorenz_coupled(; s=lorenz_time_scale, c1=lorenz_fine_coupl, c2=lorenz_crse_coupl) = OdeSystem{6}(
    "Two-time Lorenz (s=$s, c1=$c1, c2=$c2)",
    u->f_lorenz_coupled(u; p=[σ, ρ, β, s, c1, c2]),
    u->J_lorenz_coupled(u; p=[σ, ρ, β, s, c1, c2]))

# simpler coupled lorenz system

# function f_lorenz_coupled(u::AbstractVector{T}; p=[σ, ρ, β, lorenz_time_scale, lorenz_fine_coupl, lorenz_crse_coupl]) where T
#     xs, ys, zs, xf, yf, zf = u
#     σ, ρ, β, s, c1, c2 = p
#     SVector{6, T}(
#         σ*(ys - xs),
#         xs*(ρ - zs) - ys,
#         xs*ys - β*(zs + c2*zf),
#         s*σ*(yf - xf),
#         s*(xf*(ρ - zf) - yf),
#         s*(xf*yf - β*(c1*zs + zf))
#     )
# end

# function J_lorenz_coupled(u::AbstractVector{T}; p=[σ, ρ, β, lorenz_time_scale, lorenz_fine_coupl, lorenz_crse_coupl]) where T
#     u1 = SVector{3, T}(u[SVector(1, 2, 3)])
#     u2 = SVector{3, T}(u[SVector(4, 5, 6)])
#     σ, ρ, β, s, c1, c2 = p
#     J = zero(MMatrix{6, 6, T})
#     J[1:3, 1:3] .= J_lorenz(u1; p=[σ, ρ, β])
#     J[3, 6] = -c2*β
#     J[4:6, 4:6] .= s*J_lorenz(u2; p=[σ, ρ, β])
#     J[6, 3] = -s*c1*β
#     return SMatrix(J)
# end

# lorenz_coupled = OdeSystem{6}("Two-time Lorenz", f_lorenz_coupled, J_lorenz_coupled)

# Forward Euler
# function euler_step(u0, h, sys::OdeSystem)
#     @show h
#     @show u0
#     @show SVector(u0 + h*sys.f(u0))
# end

const rossler_ps = (0.25, 3.0, 0.05, 0.5)

# Rossler hyperchaos
function f_rossler(u::AbstractVector{T}; p=rossler_ps)::SVector{4, T} where T
    x, y, z, w = u
    a, b, c, d = p
    SVector{4, T}(
        -y - z,
        x + a*y + w,
        b + x*z,
        c*w - d*z
    )
end

function J_rossler(u::AbstractVector{T}; p=rossler_ps)::SMatrix{4, 4, T} where T
    x, y, z, w = u
    a, b, c, d = p
    @SMatrix T[
        0 -1 -1 0
        1  a  0 1
        z  0  x 0
        0  0 -d c
    ]
end

const rossler_hyperchaos = OdeSystem{4}("Rossler hyperchaos", f_rossler, J_rossler)

# Rikitake dynamo

function f_riki(u::AbstractVector{T})::SVector{3, T} where T
    x, y, z = u
    SVector{3, T}(
        -x + y*z,
        -y + x*(z - 1.),
        1. - x*y
    )
end

function J_riki(u::AbstractVector{T}; p=rossler_ps)::SMatrix{3, 3, T} where T
    x, y, z = u
    @SMatrix T[
         -1     z    y
        z - 1  -1    x
         -y    -x    0
    ]
end

const rikitake = OdeSystem{3}("Rikitake dynamo", f_riki, J_riki)

# lorenz emanuel system (lorenz 96)
f_le(x::AbstractVector{T}; args...) where T = f_le(SVector(x...); args...)
function f_le(x::StaticVector{N, T}; F=8.) where {N, T}

    # periodic indices
    i::SVector{N, Int} = eachindex(x)
    im2 = mod1.(i .- 2, N)
    im1 = mod1.(i .- 1, N)
    ip1 = mod1.(i .+ 1, N)

    return (x[ip1] - x[im2]) .* x[im1] - x .+ F
end

J_le(x::AbstractVector{T}; args...) where T = J_le(SVector(x...); args...)
function J_le(x::StaticVector{N, T}; F=8.) where {N, T}
    J = zero(MMatrix{N, N, T})

    for i ∈ 1:N
        im2 = mod1(i - 2, N)
        im1 = mod1(i - 1, N)
        ip1 = mod1(i + 1, N)
        J[i, im2] = -x[im1]
        J[i, im1] = x[ip1] - x[im2]
        J[i, i] = -one(T)
        J[i, ip1] = x[im1]
    end
    SMatrix(J)
end

lorenz_emanuel(N, F=8) = OdeSystem{N}("Conservative Lorenz Emanuel", x->f_le(x; F=F), x->J_le(x; F=F))

# conservative lorenz emanuel system
f_lecons(x::AbstractVector{T}) where T = f_lecons(SVector(x...))

function f_lecons(x::StaticVector{N, T}) where {N, T}
    # periodic indices
    i::SVector{N, Int} = eachindex(x)
    im2 = mod1.(i .- 2, N)
    im1 = mod1.(i .- 1, N)
    ip1 = mod1.(i .+ 1, N)

    return (x[ip1] - x[im2]) .* x[im1]
end

J_lecons(x::AbstractVector{T}) where T = J_lecons(SVector(x...))
function J_lecons(x::StaticVector{N, T}) where {N, T}
    J = zero(MMatrix{N, N, T})

    for i ∈ 1:N
        im2 = mod1(i - 2, N)
        im1 = mod1(i - 1, N)
        ip1 = mod1(i + 1, N)
        J[i, im2] = -x[im1]
        J[i, im1] = x[ip1] - x[im2]
        J[i, ip1] = x[im1]
    end
    SMatrix(J)
end

lorenz_emanuel_cons(N) = OdeSystem{N}("Conservative Lorenz Emanuel", f_lecons, J_lecons)

# this measures the energy in the le system (it should be conserved)
le_energy(x) = x'*x

# function f_lorenz(u::AbstractVector{T}; p=lorenz_ps)::SVector{3, T} where T
#     x, y, z = u
#     σ, ρ, β = p
#     SVector{3, T}(
#         σ*(y - x),
#         x*(ρ - z) - y,
#         x*y - β*z
#     )
# end

# discretized KS equation
# f_DKS(u::AbstractVector{T}) where T = f_DKS(SVector(u...))
function f_DKS(u::StaticVector{N, T}) where {N, T}
    # periodic indices
    i::SVector{N, Int} = eachindex(u)
    im2 = mod1.(i .- 2, N)
    im1 = mod1.(i .- 1, N)
    ip1 = mod1.(i .+ 1, N)
    ip2 = mod1.(i .+ 2, N)

    return u[i] .* (u[ip1] .- u[im1]) .- 5u[i] .+ 3.5*(u[ip1] .+ u[im1]) .- u[ip2] .- u[im2]
end

function f_DKS(u::AbstractVector{T}) where T
    f = similar(u)
    for i ∈ eachindex(u)
        im2 = mod1(i - 2, N)
        im1 = mod1(i - 1, N)
        ip1 = mod1(i + 1, N)
        ip2 = mod1(i + 2, N)

        f[i] = u[i]*(u[ip1] - u[im1]) - 5u[i] + 3.5*(u[ip1] + u[im1]) - (u[ip2] + u[im2])
    end
end

function J_DKS(u::AbstractVector{T}) where T
    J = zeros(T, length(u), length(u))

    for i ∈ eachindex(u)
        im2 = mod1(i - 2, N)
        im1 = mod1(i - 1, N)
        ip1 = mod1(i + 1, N)
        ip2 = mod1(i + 2, N)

        J[i, im2] = -1.
        J[i, im1] = -u[i] + 3.5
        J[i, i]   = u[ip1] - u[im1] - 5.
        J[i, ip1] = u[i] + 3.5
        J[i, ip2] = -1.
    end

    return J
end

# J_DKS(u::AbstractVector{T}) where T = J_DKS(SVector(u...))
function J_DKS(u::StaticVector{N, T}) where {N, T}
    J = zero(MMatrix{N, N, T})

    for i ∈ 1:N
        im2 = mod1(i - 2, N)
        im1 = mod1(i - 1, N)
        ip1 = mod1(i + 1, N)
        ip2 = mod1(i + 2, N)

        J[i, im2] = -1.
        J[i, im1] = -u[i] + 3.5
        J[i, i]   = u[ip1] - u[im1] - 5.
        J[i, ip1] = u[i] + 3.5
        J[i, ip2] = -1.
    end

    SMatrix(J)
end

discrete_ks(N) = OdeSystem{N}("Discrete KS", f_DKS, J_DKS)

euler_step(u::S, h, sys::OdeSystem{1}) where S = u + h*sys.f(u)
euler_step(u1::S, u0::S, h, sys::OdeSystem{1}) where S = u0 + h*sys.f(u0)
euler_step_du(u::S, h, sys::OdeSystem{1}) where S = one(S) + h*sys.J(u)
euler_step_dt(u::S, h, sys::OdeSystem{1}) where S = sys.f(u)
euler_step_Hv(v::S, u::S, h, sys::OdeSystem{1}) where S = [
    h*sys.Hv(v, u) sys.J(u)*v
    sys.J(u)*v     zero(S)
]

function euler_step(u0::AbstractArray{T}, h, sys::OdeSystem{N})::SVector{N, T} where {N, T}
    u0 + h*sys.f(u0)
end

function euler_step(u1::AbstractArray{T}, u0::AbstractArray{T}, h, sys::OdeSystem{N})::SVector{N, T} where {N, T}
    u0 + h*sys.f(u0)
end

function euler_step_du(u0, h, sys::OdeSystem{N}) where {N}
    I(N) .+ h*sys.J(u0)
end

function euler_step_du(u0::ST, h, sys::OdeSystem{N})::SMatrix{N, N, T} where {N, ST <: Union{SVector{N, T}, MVector{N, T}}} where T
    SDiagonal{N, T}(I) + h*sys.J(u0)
end

function euler_step_dt(u0, h, sys::OdeSystem)
    sys.f(u0)
end

# implements ∂/∂(u,h)(∂Φ/∂(u, h)^T v) ∈ ℝ^{N+1, N+1}
function euler_step_Hv(v::AbstractVector{T}, u0, h, sys::OdeSystem) where T
    dhdu = sys.J(u0)'*v
    [
        [h*sys.Hv(v, u0) dhdu   ]
        [dhdu'           zero(T)]
    ]
end

euler_step_withgrads(u0, args...) = euler_step(u0, args...), (euler_step_du(u0, args...), euler_step_dt(u0, args...))

# 1st order θ-method
# u1 = u0 + θhf(u0) + (1-θ)hf(u1)
θ_res(u1, ps) = θ_res(u1, ps[1], ps[2], ps[3]; θ=ps[4])
function θ_res(u1::AbstractVector{T1}, u0::AbstractVector{T2}, h, sys::OdeSystem{N}; θ=1/2) where {N, T1, T2}
    return u1 - u0 - θ*h*sys.f(u0) - (1 - θ)*h*sys.f(u1)
end

θ_res_du1(u1, ps) = θ_res_du1(u1, ps...; θ=1/2)
function θ_res_du1(u1::V, u0::V, h, sys::OdeSystem{N}; θ=1/2) where {N, V <: SVector{N, T}} where T
    return SDiagonal{N, T}(I) .- (1 - θ)*h*sys.J(u1)
end

function θ_res_du1(u1, u0, h, sys::OdeSystem{N}; θ=1/2) where N
    return I(N) .- (1 - θ)*h*sys.J(u1)
end

function θ_res_du(u1, u0, h, sys::OdeSystem{N}; θ=1/2) where {N}
    (I(N) .- (1-θ)*h*sys.J(u1), -I(N) .- θ*h*sys.J(u0))
end

function θ_res_du(u1::V, u0::V, h, sys::OdeSystem{N}; θ=1/2) where {N, V <: SVector{N, T}} where T
    (SDiagonal{N, T}(I) - (1-θ)*h*sys.J(u1), -SDiagonal{N, T}(I) - θ*h*sys.J(u0))
end

function θ_res_dt(u1, u0, h, sys::OdeSystem{N}; θ=1/2) where N
    -(1-θ)*sys.f(u1) - θ*sys.f(u0)
end

function θ_step(u1::AbstractVector{T}, u0, h, sys::OdeSystem{N}; θ=1/2) where {N, T}
    ps = (u0, h, sys, θ)
    prob = NonlinearProblem(θ_res, u1, ps)
    # return my_newton(θ_res, u1, ps; reltol=1e-3)
    return solve(prob, SimpleNewtonRaphson()).u
end

function θ_step(u1::AbstractVector{T}, u0, h, sys::OdeSystem{N}, τ; θ=1/2) where {N, T}
    ps = (u0, h, sys, θ)
    prob = NonlinearProblem((u, p)->τ - θ_res(u, p), u1, ps)
    # return my_newton(θ_res, u1, ps; reltol=1e-3)
    return solve(prob, SimpleNewtonRaphson()).u
end

function θ_step_differentiable(u0::AbstractArray{T1}, h::T2, sys::OdeSystem{N}; θ=1/2) where {N, T1, T2}
    u0 = SVector{N, promote_type(T1, T2)}(u0)
    ps = (u0, h, sys, θ)
    return my_newton(θ_res, u0, ps; abstol=1e-10, reltol=1e-10)
end
θ_step_differentiable(u1, u0, h, sys; θ=1/2) = θ_step_differentiable(u0, h, sys; θ=θ)

function θ_step_du(u0, h, sys::OdeSystem; θ=1/2)
    u1 = θ_step_differentiable(u0, u0, h, sys)
    J1, J0 = θ_res_du(u1, u0, h, sys)
    return -J1 \ J0
end

function θ_step_dt(u0, h, sys::OdeSystem; θ=1/2)
    u1 = θ_step_differentiable(u0, u0, h, sys)
    J1 = θ_res_du1(u1, u0, h, sys)
    return J1\(θ*sys.f(u0) + (1-θ)*sys.f(u1))
end

const euler = TimeStepper("Forward Euler", euler_step, euler_step_du, euler_step_dt, euler_step_Hv, euler_step_withgrads)
# theta_method = TimeStepper("θ method (explicit form)", θ_step_differentiable, θ_step_du, θ_step_dt, nothing, nothing)
function theta_method(θ)
    Φ = (args...)->θ_step_differentiable(args...; θ=θ)
    Φ_du = (args...)->θ_step_du(args...; θ=1/2)
    Φ_dt = (args...)->θ_step_dt(args...; θ=1/2)
    return TimeStepper("θ-method (θ = $θ)", Φ, Φ_du, Φ_dt, nothing, nothing)
end

# function imp_theta_method(θ)
#     Φ = (args...)->θ_step(args...; θ=θ)
#     res = (args...)->θ_res(args...; θ=θ)
#     res_du = (args...)->θ_res_du(args...; θ=θ)
#     res_dt = (args...)->θ_res_dt(args...; θ=θ)
#     return ImplicitTimeStepper("θ-method (θ = $θ)", Φ, res, res_du, res_dt, nothing)
# end

# test_all()
