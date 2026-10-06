using LinearAlgebra, NonlinearSolve
import Base.Threads.@threads

struct lsr_NLparams{TS<:AbstractTimeDiscretization, OS<:AbstractDiffEQ, UT1, VT, UT2, R}
    u::UT1
    η::VT
    τ::UT2
    Δt::R
    stp::TS
    sys::OS
    α::R
    γ::R
    lsr_NLparams(u::UT1, η::VT, τ::UT2, Δt::R, stp::TS, sys::OS, α; γ=1.) where {TS, OS, UT1, VT, UT2, R}  = new{TS, OS, UT1, VT, UT2, R}(u, η, τ, Δt, stp, sys, α, γ)
end

# compute dilation η = dilate(control)
# dilate(c::T) where T = one(T) + tanh(c)
# dilate_dη(c::T) where T = (one(T) - tanh(c)^2)
# dilate_inv(η::T) where T = atanh(η - one(T))
dilate(c::T) where T = abs(one(T) + c)
dilate_dη(c::T) where T = (one(T) + c)/abs(one(T) + c)
# dilate(c::T) where T = one(T) + c
# dilate_dη(c::T) where T = one(T)
dilate_inv(η::T) where T = η - one(T)

dilate(c, Δt::T) where T = dilate(c)*Δt
dilate_dη(c, Δt::T) where T = dilate_dη(c)*Δt

# f- and c-relaxation
f_step(u1, u0, η, h, stp::AbstractTimeDiscretization, sys) = stp.Φ(u1, u0, dilate(η, h), sys)
f_step(u1, u0, η, h, stp::AbstractTimeDiscretization, sys, τ) = stp.Φ(u1, u0, dilate(η, h), sys) .+ τ

function f_block!(u::AbstractArray{S, D}, η::AbstractVector{T}, τ, Δt::T, disc::AbstractTimeDiscretization, sys::AbstractDiffEQ, block_inds::AbstractRange; kwargs...) where {T, S, D}
    for j ∈ block_inds
        time_slice(u, j+1) .= f_step(time_slice(u, j+1), time_slice(u, j), η[j], Δt, disc, sys, time_slice(τ, j))
    end
end

function f_block!(u::AbstractArray{<:SArray, 1}, η::AbstractVector, τ::AbstractArray{<:SArray, 1}, Δt::Real, disc::AbstractTimeDiscretization, sys::AbstractDiffEQ, block_inds::AbstractRange; kwargs...)
    for j ∈ block_inds
        u[j+1] = f_step(u[j+1], u[j], η[j], Δt, disc, sys, τ[j])
    end
end

function f_relax!(u, η, τ, m, Δt, stp::AbstractTimeDiscretization, sys::AbstractDiffEQ)
    Nt = num_time_steps(u)

    # for i ∈ reverse(c_points(Nt, m, TimeCentered()))
    @threads for i ∈ reverse(c_points(Nt, m, TimeCentered()))
        block_inds = i:min(i+m-2, Nt)
        f_block!(u, η, τ, Δt, stp, sys, block_inds)
    end
    return
end


function c_relax!(u::AbstractArray{S, D}, η::AbstractVector{T}, τ::AbstractArray{S, D}, m::Integer, Δt::T, stp::AbstractTimeDiscretization, sys::AbstractDiffEQ) where {T, S, D}
    Nt = num_time_steps(u)

    @threads for i ∈ reverse(c_points(Nt, m, TimeCentered())[2:end])
        selectdim(u, D, i) .= f_step(selectdim(u, D, i), selectdim(u, D, i-1), η[i-1], Δt, stp, sys, selectdim(τ, D, i-1))
    end
    return
end

function c_relax!(u::AbstractVector{SVector{N, S}}, η::AbstractVector, τ::AbstractVector{SVector{N, S}}, m::Integer, Δt::Real, stp::AbstractTimeDiscretization, sys::AbstractDiffEQ) where {S, N}
    Nt = num_time_steps(u)

    @threads for i ∈ c_points(Nt, m, TimeCentered())[2:end]
        @inbounds u[i] = f_step(u[i], u[i-1], η[i-1], Δt, stp, sys, τ[i-1])
    end
    return
end

function lsr_residual(p::AbstractVector{R}, u::AbstractVector{SVector{N, T}}, η::AbstractVector, τ, Δt, step::TimeStepper, sys::OdeSystem{N}) where {N, T, R}
    m = length(u) - 1

    p_u = SVector(pop(p))
    p_η = p[end]

    r = zero(MVector{N, R})
    r .= u[1] .+ p_u
    dΦdη = zero(MVector{N, R})
    dΦdp = MMatrix{N, N, R}(I)
    for i = 1:m
        Φ_u = step.Φ_u(SVector(r), dilate(η[i] + p_η, Δt), sys)
        Φ_η = dilate_dη(η[i] + p_η, Δt)*step.Φ_t(SVector(r), dilate(η[i] + p_η, Δt), sys)
        r .= step.Φ(SVector(r), SVector(r), dilate(η[i] + p_η, Δt), sys) .+ τ[i]
        dΦdp .= Φ_u*dΦdp
        dΦdη .= Φ_u*dΦdη .+ Φ_η
    end
    r .= r .- u[end]

    return SVector(r), SMatrix(dΦdp), SVector(dΦdη)
end

# This is super slow when m > ~8
# function lsr_residual(p::SVector{NP, T}, u::AbstractVector{SVector{N, T}}, η::AbstractVector, τ, Δt, disc::ImplicitTimeStepper, sys::OdeSystem{N}) where {NP, N, T}
#     pη = p[end]

#     NM = NP - 1 - N
#     M = NM ÷ N
#     r = MVector{NM, T}(undef)
#     A = zero(MMatrix{NM, NP})

#     block = Block{N}()

#     for i = 1:M
#         block(r, i) .= τ[i] - disc.residual(u[i+1] + p[block(i+1)], u[i] + p[block(i)], dilate(η[i]+pη, Δt), sys)
#         J1, J0 = disc.residual_J(u[i+1] + p[block(i+1)], u[i] + p[block(i)], dilate(η[i]+pη, Δt), sys)
#         Jη = dilate_dη(η[i]+pη, Δt)*disc.residual_dt(u[i+1] + p[block(i+1)], u[i] + p[block(i)], dilate(η[i]+pη, Δt), sys)
#         block(A, i, i) .= -J0
#         block(A, i, i+1) .= -J1
#         @views A[block(i), end] .= -Jη
#     end

#     return SVector(r), SMatrix(A)
# end

# struct Block{N} <: Function 
#     Block{N}() where N = new{N}()
#     Block(::Val{N}) where N = new{N}()
# end
# # (::Block{N})(i::I) where {I<:Integer, N} = SVector{N, I}(UnitRange{I}((i-1)*N+1, i*N))
# (::Block{N})(i::I) where {I<:Integer, N} = SVector{N, I}((i-1)*N+1:i*N)
# (b::Block)(r::AbstractRange) = vcat((b(i) for i ∈ r)...)
# (b::Block)(c) = map(b, c)
# (b::Block)(A::AbstractArray, i::Number) = view(A, b(i))
# @inline (b::Block)(A::AbstractArray, i::Number, j::Number) = view(A, b(i), b(j))
# @inline (b::Block)(A::AbstractArray, inds...) = view(A, b(inds)...)

# function lsr_residual(p::AbstractVector{T}, u::AbstractVector{SVector{N, T}}, η, τ, Δt, disc::ImplicitTimeStepper, sys::OdeSystem{N}) where {N, T}
#     pη = p[end]

#     M = length(u)-1
#     r = similar(p, N*M)
#     A = similar(p, (N*M, length(p)))
#     fill!(A, zero(T))

#     block = Block{N}()

#     for i = 1:M
#         block(r, i) .= τ[i] - disc.residual(u[i+1] + block(p, i+1), u[i] + block(p, i), dilate(η[i]+pη, Δt), sys)
#         J1, J0 = disc.residual_J(u[i+1] + block(p, i+1), u[i] + block(p, i), dilate(η[i]+pη, Δt), sys)
#         Jη = dilate_dη(η[i]+pη, Δt)*disc.residual_dt(u[i+1] + block(p, i+1), u[i] + block(p, i), dilate(η[i]+pη, Δt), sys)
#         block(A, i, i) .= -J0
#         block(A, i, i+1) .= -J1
#         @views A[block(i), end] .= - Jη
#     end

#     return r, A
# end

lsr_objective(r, p, η; α=1.) = 1/2 * (norm(r)^2 + norm(p[1:3])^2) + α^2/2 * p[end]^2
lsr_objective(p, u, η, τ, Δt, step, sys; α=1.) = lsr_objective(lsr_residual(p, u, η, τ, Δt, step, sys)[1], p, η; α=α)

function lsr_gauss_newton_res(p::SVector{N, T}, ps::lsr_NLparams{<:TimeStepper}) where {N, T}
    r_u, Ju, Jη = lsr_residual(p, ps)

    r_p = MVector(p)
    # r_p[end] = ps.α*(r_p[end] + sum(ps.η)/length(ps.η))
    r_p[end] *= ps.α
    L = [ones(SVector{N-1, T}); ps.α]

    return [r_u; r_p], [[Ju Jη]; SDiagonal{N, T}(L)]
end

function lsr_g(p, u, η, τ, Δt, step, sys; α=1.)
    p_u = SVector(pop(p))
    p_η = p[end]

    r, dΦdp, dΦdη = lsr_residual(p, u, η, τ, Δt, step, sys)
    g_u = p_u .+ dΦdp' * r
    g_η = α^2*p_η + dΦdη' * r

    g = vcat(g_u, g_η)
    return g
end

function lsr_residual(p, ps::lsr_NLparams)
    lsr_residual(p, ps.u, ps.η, ps.τ, ps.Δt, ps.stp, ps.sys)
end

function lsr_g(p, ps::lsr_NLparams)
    lsr_g(p, ps.u, ps.η, ps.τ, ps.Δt, ps.stp, ps.sys; α=ps.α)
end

struct Counting{TF} <: Function
    f::TF
    counter::Ref{Int}
end
Counting(f, c::Integer) = Counting(f, Ref{Int}(c))

Counting(f) = Counting(f, 0)
function (c::Counting)(args...)
    c.counter[] += 1
    c.f(args...)
end

function lsr_block!(u::AbstractVector{SVector{N, T}},
                    η::AbstractVector,
                    τ::AbstractVector{SVector{N, T}},
                    Δt::Real, step::TimeStepper,
                    sys::OdeSystem{N};
                    alg=:my_gauss_newton,
                    α=1., finalFC=true,
                    damping=1.,
                    solver_kwargs...) where {N, T}
    m = length(u) - 1
    m > 0 || return

    p0 = zero(SVector{N+1, T})
    ps = lsr_NLparams(u, η, τ, Δt, step, sys, α)

    # try
        if alg == :my_newton
            sol = my_newton(lsr_g, p0, ps; solver_kwargs...)
        elseif alg == :my_gauss_newton
            sol = my_gauss_newton(lsr_gauss_newton_res, p0, ps; solver_kwargs...)
        elseif alg == :my_gauss_newton_cons
            cons = similar(p0)
            r, dΦdu, dΦdh = lsr_residual(p0, ps)
            # cons[1:N] .= dΦdu' * dΦdh
            # cons[end] = dΦdh' * dΦdh
            # cons_rhs = -dΦdh' * r
            # cons[1:N] .= dΦdh
            # cons[1:N] .= dΦdu \ dΦdh .+ 100randn(3)
            cons[1:N] .= dΦdu \ dΦdh
            # cons[1:N] .= 100rand(3)
            # cons[1:N] .= dΦdu \ dΦdh
            # cons[1:N] .= dΦdu' * dΦdh
            # cons[end] = zero(T)
            cons_rhs = zero(T)
            sol = my_gauss_newton(lsr_gauss_newton_res, (SVector(cons), cons_rhs), p0, ps; solver_kwargs...)
        else
            prob = NonlinearProblem{false}(lsr_g, p0, ps)
            sol = solve(prob, alg; solver_kwargs...).u
        end
        η .+= sol[end]
        u[1] += damping*pop(sol)
    # catch SingularException
    #     p0 = zero(SVector{N+1, T})
    #     prob = NonlinearProblem{false}(lsr_g, p0, ps)
    #     sol = solve(prob, SimpleBroyden(); reltol=1e-2).u
    #     η .+= sol[end]
    #     u[1] += pop(sol)
    # end

    if finalFC
        for i = 1:m
            u[i+1] = step.Φ(u[i+1], u[i], dilate(η[i], Δt), sys) .+ τ[i]
        end
    end
    return
end

# function lsr_block!(u::AbstractVector{SVector{N, T}},
#                     η::AbstractVector,
#                     τ::AbstractVector{SVector{N, T}},
#                     Δt::Real, step::ImplicitTimeStepper,
#                     sys::OdeSystem{N};
#                     finalFC=true,
#                     underdetermined=false,
#                     α=1., γ=1.0, solver_args...) where {N, T}
#     # use Gauss-Newton method to solve KKT system of linearized constrained optimization
#     m = length(u)-1
#     m > 0 || return

#     np = (m+1)*N + 1
#     block = Block{N}()

#     ps = lsr_NLparams(u, η, τ, Δt, step, sys, α; γ=γ)

#     d = zero(MVector{np, T})
#     block(d, 1) .= γ
#     block(d, m+1) .= γ

#     d[end] = γ*α
#     d = SVector(d)


#     if m <= 8
#         W = SDiagonal(d)
#         p = zero(SVector{np, T})
#     else
#         W = Vector(d)
#         p = zeros(T, np)
#     end
#     pdamp = zero(MVector{np, T})
#     p = KKT_gauss_newton(W, lsr_residual, p, ps; pdamp=SVector(pdamp), solver_args...)

#     η .+= p[end]

#     for i ∈ 1:m+1
#         u[i] += p[block(i)]
#     end
#     return
# end

@inline function lsr_block!(u::AbstractArray{S, D}, η::AbstractVector{T}, τ, Δt::T, disc::AbstractTimeDiscretization, sys::AbstractDiffEQ, block_inds::AbstractRange; kwargs...) where {T, S, D}
    lsr_block!(selectdim(u, D, block_inds), view(η, block_inds[1:end-1]), selectdim(τ, D, block_inds[1:end-1]), Δt, disc, sys; kwargs...)
end

@inline function lsr_block!(u::AbstractArray{<:SArray, 1}, η::AbstractVector, τ::AbstractArray{<:SArray, 1}, Δt::Real, disc::AbstractTimeDiscretization, sys::AbstractDiffEQ, block_inds::AbstractRange; kwargs...)
    @views lsr_block!(u[block_inds], η[block_inds[1:end-1]], τ[block_inds[1:end-1]], Δt, disc, sys; kwargs...)
end

function lsr!(u, η, τ, m, Δt, disc::AbstractTimeDiscretization, sys::AbstractDiffEQ; iters::Integer=1, overlap::Integer=0, overrelax::Integer=1, skip_finalFC=false, f_relax=true, initial_boundary=false, kwargs...)
    Nt = num_time_steps(u)
    # multiplying m by an integer increases the span of relaxation on each level
    mf = m
    m = m*overrelax

    cpoints = c_points(Nt, m, TimeCentered())
    @assert overlap <= m÷2-1
    if (m%2 == 1)
        overlap += 1
    end

    # initial F-relax
    # if overrelax == 1
        # f_relax!(u, η, τ, mf, Δt, disc, sys)
        # c_relax!(u, η, τ, mf, Δt, disc, sys)

    # end

    for k = 1:iters
        # C-lsr
        @threads for i ∈ reverse(cpoints)
        # for i ∈ reverse(cpoints)
            block = max(1, i - m÷2 - overlap):i
            lsr_block!(u, η, τ, Δt, disc, sys, block; kwargs...)
        end

        # F-lsr
        @threads for i ∈ reverse(cpoints)
        # for i ∈ reverse(cpoints)
            block = i:min(i + m÷2 + overlap, Nt+1)
            if initial_boundary && (i == 1)
                f_block!(u, η, τ, Δt, disc, sys, block)
            else
                lsr_block!(u, η, τ, Δt, disc, sys, block; finalFC=!(skip_finalFC && k==iters), kwargs...)
            end
        end
    end

    # final F-relax
    if f_relax
        f_relax!(u, η, τ, mf, Δt, disc, sys)
    end
    # fcf_relax!(u, η, τ, mf, Δt, disc, sys)
end

# function lsr_block!(u::AbstractArray{S, D},
#                     η::AbstractVector{T},
#                     τ::AbstractArray{S, D},
#                     Δt::T, step::AbstractTimeDiscretization,
#                     sys::OdeSystem{N};
#                     alg=NewtonRaphson(),
#                     α=1., finalFC=true,
#                     solver_kwargs...) where {T, S, D, N}
#     m = size(u, D) - 1
#     m > 0 || return

#     TS = promote_type(T, S)

#     p0 = zeros(TS, N+1)
#     ps = lsr_NLparams(u, η, τ, Δt, step, sys, α)

#     prob = NonlinearProblem{false}(lsr_g, p0, ps)
#     sol = solve(prob, alg; solver_kwargs...).u

#     @views η .+= sol[end]
#     selectdim(u, D, 1) .+= view(sol, 1:N)

#     if finalFC
#         for i = 1:m
#             # u_{i+1} = Φ(u_{i+1}, u_i, (1 + η_i)h) + τ_i,              i = 1, 2, ..., m
#             selectdim(u, D, i+1) .= step.Φ(selectdim(u, D, i+1), selectdim(u, D, i), (one(T) + η[i])*Δt, sys) .+ selectdim(τ, D, i)
#         end
#     end
#     return
# end