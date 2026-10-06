include("methods.jl")
include("lsr.jl")

# Here I'm assuming everything is 1-based indexed.
# State vectors (u) time-centered: indexed based on time points (Nt+1), with initial condition at index 1.
# Other vectors (τ, r, η...) are step-centered: indexed based on time steps (Nt), such that e.g.
# 
# u_{i+1} = Φ(u_{i+1}, u_i, (1 + η_i)h) + τ_i,              i = 1, 2, ..., Nt
# A_0     = u_1
# A_i     = u_{i+1} - Φ(u_{i+1}, u_i, (1 + η_i)h)           i = 1, 2, ..., Nt
# r_i     = τ_i - u_{i+1} + Φ(u_{i+1}, u_i, (1 + η_i)h)
# τ_ic    = rf_ic - rc_{m(ic)} 
#         = Φ(u_{m(ic)+1}, u_{m(ic)}, (1 + η_{m(ic)})h) - Φ(u_{ic+1}, u_ic, m(1 + ηc_ic)h),      ic = 1, 2, ..., Nt÷m
# 
# ηc_{ic} = 1/m ∑_{i ∈ (m(ic-1)+1):(m(ic)) }(η_i)     ic = 1, 2, ..., Nt÷m
# 
#  ∘ = C-point
#  ⋅ = F-point
#     ∘       ∘       ∘    // ∘           ∘           ∘
# uc: 1       2       3...    ic-1        ic          ic+1
# τc:         1       2...    ic-2        ic-1        ic  
# ηc:         1       2...    ic-2        ic-1        ic  
#     ∘ ⋅ ⋅ ⋅ ∘ ⋅ ⋅ ⋅ ∘    // ∘  ⋅  ⋅  ⋅  ∘  ⋅  ⋅  ⋅  ∘  ⋅  ⋅  ⋅  
#  u: 1 2 3 4 5 6 7 8 9...    m(ic-2)+1   m(ic-1)+1   m(ic)+1      ic = 1, 2, ..., Nt÷m+1
#  τ:   1 2 3 4 5 6 7 8...    m(ic-2)     m(ic-1)     m(ic)        ic = 1, 2, ..., Nt÷m
#  η:   1 2 3 4 5 6 7 8...    m(ic-2)     m(ic-1)     m(ic)        ic = 1, 2, ..., Nt÷m

struct TimeGrid{T, S, D, OS<:AbstractDiffEQ, TS<:AbstractTimeDiscretization, U<:AbstractArray{S, D}, G<:AbstractArray{S, D}, H<:AbstractVector{T}}
    u::U  # state vector
    r::Union{U, Nothing}  # restricted fine state
    τ::G    # forcing vector
    η::H  # time dilation
    ηr::Union{H, Nothing} # restricted time dilation
    sys::OS # spatial discretization
    disc::TS # temporal discretization
    Δt::T   # time step size
    m::Int  # coarsening factor
    Nt::Int # number of time steps (== time points - 1)
    # spatial coarsening functions are owned by the coarse grid 
    # e.g. fgrid.u .= cgrid.srefine(cgrid.u))
    #      cgrid.u .= cgrid.scoarsen(fgrid.u))
    ssize_coarsen # returns coarse spatial size given fine
    sctype   # returns coarse spatial eltype given fine (e.g. sctype(::Type{SVector{6, Float64}) = SVector{3, Float32})
    scoarsen # spatial coarsening routine
    srefine  # spatial refinement routine
    function TimeGrid(u::U, r::U, τ::G, η::H, ηr::H, sys::OS, disc::TS, Δt::T, m::Integer; scoarsen::Function=identity, srefine::Function=identity, ssize_coarsen::Function=identity, sctype::Function=Returns(S)) where {U<:AbstractArray{S, D}, G<:AbstractArray{S, D}, H<:AbstractVector{T}, OS, TS} where {T, S, D}
        Nt = num_time_steps(u)
        @assert size(τ, D) == Nt
        @assert size(r, D) == Nt+1
        @assert size(η, 1) == Nt
        @assert size(ηr, 1) == Nt
        new{T, S, D, OS, TS, U, G, H}(u, r, τ, η, ηr, sys, disc, Δt, m, Nt, ssize_coarsen, sctype, scoarsen, srefine)
    end
    function TimeGrid(u::U, r::Nothing, τ::G, η::H, ηr::Nothing, sys::OS, disc::TS, Δt::T, m::Integer; scoarsen::Function=identity, srefine::Function=identity, ssize_coarsen::Function=identity, sctype::Function=Returns(S)) where {U<:AbstractArray{S, D}, G<:AbstractArray{S, D}, H<:AbstractVector{T}, OS, TS} where {T, S, D}
        Nt = num_time_steps(u)
        @assert size(τ, D) == Nt
        @assert size(η, 1) == Nt
        new{T, S, D, OS, TS, U, G, H}(u, r, τ, η, ηr, sys, disc, Δt, m, Nt, ssize_coarsen, sctype, scoarsen, srefine)
    end
end

TimeGrid(u, τ, η, sys, disc, Δt, m; scargs...) = TimeGrid(u, nothing, τ, η, nothing, sys, disc, Δt, m; scargs...)
TimeGrid(u::AbstractArray{S, D}, sys, disc, Δt::T, m; scargs...) where {S, D, T} = TimeGrid(u,
                                                                           fill(zero(S), (size(u)[1:D-1]..., num_time_steps(u))),
                                                                           zeros(T, num_time_steps(u)),
                                                                           sys, disc, Δt, m; scargs...
                                                                           )

abstract type Centering end
struct TimeCentered <: Centering end
struct StepCentered <: Centering end

# helper functions for using point-centered and step-centered arrays
num_time_steps(u::AbstractArray{S, D}) where {S, D}  = size(u, D)-1 # assume time dimension is last
num_time_steps(g::TimeGrid{T, S, D}) where {T, S, D} = g.Nt
num_time_points(u::AbstractArray{S, D}) where {S, D}  = size(u, D) # assume time dimension is last
num_time_points(g::TimeGrid{T, S, D}) where {T, S, D} = g.Nt+1
f_index(ic, m, ::TimeCentered) = m*(ic-1)+1
f_index(ic, m, ::StepCentered) = m*ic
c_points(Nt::Integer, m::Integer, ::TimeCentered) = 1:m:Nt+1
c_points(Nt::Integer, m::Integer, ::StepCentered) = m:m:Nt
c_points(grid::TimeGrid, c::Centering) = c_points(grid.Nt, grid.m, c)

time_slice(u::AbstractArray{S, D}, t) where {S, D} = selectdim(u, D, t)

# norms
map_spacenorm(u::Vector{SVector{N, T}}, l=2) where {N, T} = norm.(u, l)
map_spacenorm(u::AbstractArray{T, D}, l=2) where {T, D} = mapslices(x->norm(x, l), u, dims=1:D-1)
spacetimenorm(u, time=Inf, space=2) = norm(map_spacenorm(u, space), time)

# system residual
@inline function residual(u1, u0, η::T, Δt::T, step::AbstractTimeDiscretization, sys::AbstractDiffEQ) where T
    u1 .- step.Φ(u0, dilate(η, Δt), sys)
end

function residual!(r, u1, u0, η::T, Δt::T, step::AbstractTimeDiscretization, sys::AbstractDiffEQ) where T
    r .= u1 .- step.Φ(u0, dilate(η, Δt), sys)
end

# @inline function residual(u1, u0, η::T, Δt::T, step::ImplicitTimeStepper, sys::AbstractDiffEQ) where {N, T}
    # step.residual(u1, u0, dilate(η, Δt), sys)
# end

# @inline function residual_dη(u1, u0, η::T, Δt::T, step::TimeStepper, sys::AbstractDiffEQ) where T
#     -step.Φ_t(u0, dilate(η, Δt), sys)*dilate_dη(η, Δt)
# end

# @inline function residual_dη(u1, u0, η::T, Δt::T, step::ImplicitTimeStepper, sys::AbstractDiffEQ) where T
#     step.residual_dt(u1, u0, dilate(η, Δt), sys)*dilate_dη(η, Δt)
# end

function residual(u::AbstractArray{S, D}, η::AbstractVector{T}, τ::AbstractArray{S, D}, Δt::T, step::AbstractTimeDiscretization, sys::AbstractDiffEQ) where {T, S, D}
    nt = num_time_steps(u)
    r  = similar(u, (size(u)[1:D-1]..., nt))
    for i ∈ 1:nt
        time_slice(r, i) .= time_slice(τ, i) .- residual(time_slice(u, i+1), time_slice(u, i), η[i], Δt, step, sys)
    end
    
    return r
end

@inline function residual(u::AbstractVector{SVector{N, S}}, η::AbstractVector, τ::AbstractVector{SVector{N, S}}, Δt::Real, step::AbstractTimeDiscretization, sys::AbstractDiffEQ) where {S, N}
    r = similar(u, length(u)-1)
    for i ∈ eachindex(r)
        r[i] = τ[i] - residual(u[i+1], u[i], η[i], Δt, step, sys)
    end

    return r
end

@inline residual(g::TimeGrid) = residual(g.u, g.η, g.τ, g.Δt, g.disc, g.sys)
@inline residual(u, Δt, step, sys) = residual(u, zeros(typeof(Δt), num_time_steps(u)), zero(u), Δt, step, sys) # for debugging

rel_residual(u::AbstractArray{T, D}, η, τ, Δt, disc, sys) where {T, D} = residual(u, η, τ, Δt, disc, sys) ./ reshape(map_spacenorm(time_slice(u, 2:num_time_points(u))), (ones(Int, D-1)..., num_time_steps(u)))
@inline rel_residual(g::TimeGrid) = rel_residual(g.u, g.η, g.τ, g.Δt, g.disc, g.sys)

# function solve!(istart::Integer, istop::Integer, u::AbstractVector{SVector{N, S}}, τ::, η::AbstractVector{T}, Δt::T, stp::AbstractTimeDiscretization, sys::AbstractDiffEQ) where {T, S, D, N}
#     @assert istop <= num_time_steps(u) + 1
#     for i ∈ istart:istop-1
#         selectdim(u, D, i+1) .= stp.Φ(selectdim(u, D, i+1), selectdim(u, D, i), dilate(η[i], Δt), sys) .+ selectdim(τ, D, i)
#     end
# end

@inline f_relax!(g::TimeGrid, m=g.m; kwargs...) = f_relax!(g.u, g.η, g.τ, m, g.Δt, g.disc, g.sys; kwargs...)
@inline c_relax!(g::TimeGrid, m=g.m; kwargs...) = c_relax!(g.u, g.η, g.τ, m, g.Δt, g.disc, g.sys; kwargs...)

function fcf_relax!(args...; iters=1, kwargs...)
    f_relax!(args...; kwargs...)
    for i ∈ 1:iters
        c_relax!(args...; kwargs...)
        f_relax!(args...; kwargs...)
    end
    return
end

@inline sequential_solve!(g::TimeGrid; kwargs...) = f_relax!(g.u, g.η, g.τ, g.Nt+1, g.Δt, g.disc, g.sys; kwargs...)
@inline lsr!(g::TimeGrid; kwargs...) = lsr!(g.u, g.η, g.τ, g.m, g.Δt, g.disc, g.sys; kwargs...)
lsr!(g::TimeGrid, m=g.m; kwargs...) = lsr!(g.u, g.η, g.τ, g.m, g.Δt, g.disc, g.sys; kwargs...)

function multiscale_lsr!(ml, v, η, τ, m, h, stp, sys, m0=m; args...)
    for l ∈ cat(1:ml, ml-1:-1:1; dims=1)
        lsr!(v, η, τ, m0*m^(l-1), h, stp, sys; args...)
    end
end

multiscale_lsr!(ml, grid::TimeGrid, m0=grid.m; args...) = multiscale_lsr!(ml, grid.u, grid.η, grid.τ, grid.m, grid.Δt, grid.disc, grid.sys, m0; args...)

# create coarse from fine

coarsen(::AbstractArray{S, D}, ssize::NTuple{N, <:Integer}, tsize::Integer, sctype::Function) where {S, D, N} = fill(zero(sctype(S)), (ssize..., tsize))

function coarsen(u::AbstractArray{S, D}, m::Integer, ::TimeCentered, ssize_coarsen::Function=identity, sctype::Function=Returns(S)) where {S, D} 
    coarsen(u, ssize_coarsen(size(u)[1:D-1]), num_time_steps(u)÷m+1, sctype)
end

function coarsen(v::AbstractArray{S, D}, m::Integer, ::StepCentered, ssize_coarsen::Function=identity, sctype::Function=Returns(S)) where {S, D} 
    coarsen(v, ssize_coarsen(size(v)[1:D-1]), num_time_points(v)÷m, sctype)
end

function coarsen(grid::TimeGrid{T, S}, sys::AbstractDiffEQ, disc::AbstractTimeDiscretization, mc::Integer; ssize_coarsen=identity, sctype=Returns(S), scargs...) where {T, S}
    m = grid.m
    uc = coarsen(grid.u, m, TimeCentered(), ssize_coarsen, sctype)
    rc = coarsen(grid.u, m, TimeCentered(), ssize_coarsen, sctype)
    τc = coarsen(grid.τ, m, StepCentered(), ssize_coarsen, sctype)
    ηc = coarsen(grid.η, m, StepCentered())
    ηrc = coarsen(grid.η, m, StepCentered())

    return TimeGrid(uc, rc, τc, ηc, ηrc, sys, disc, m*grid.Δt, mc; ssize_coarsen=ssize_coarsen, sctype=sctype, scargs...)
end
coarsen(grid::TimeGrid) = coarsen(grid, grid.sys, grid.disc, grid.m)

function construct_hierarchy(finegrid::TimeGrid{T, S, D}, nl::Integer;
                             systems::NTuple{NL, <:AbstractDiffEQ} = ntuple(Returns(finegrid.sys), nl-1), 
                             discs::NTuple{NL, <:AbstractTimeDiscretization} = ntuple(Returns(finegrid.disc), nl-1), 
                             cfs::NTuple{NL, <:Integer} = ntuple(Returns(finegrid.m), nl-1)
                             ) where {T, S, D, NL}
    grids = [finegrid]
    for l ∈ 2:nl
        push!(grids, coarsen(grids[l-1], systems[l-1], discs[l-1], cfs[l-1]))
        # copy initial condition
        selectdim(grids[l].u, D, 1) .= selectdim(grids[l-1].u, D, 1)
    end
    return (grids...,)
end

function restrict!(gridf::TimeGrid{T, SF, D}, gridc::TimeGrid{T, SC, D}; skip=false) where {T, SF, SC, D}
    nf = num_time_steps(gridf)
    nc = num_time_steps(gridc)
    m = gridf.m
    gridc.u .= gridc.scoarsen(selectdim(gridf.u, D, c_points(gridf, TimeCentered())))
    gridc.r .= gridc.scoarsen(selectdim(gridf.u, D, c_points(gridf, TimeCentered())))
    gridc.τ .= gridc.scoarsen(selectdim(gridf.τ, D, c_points(gridf, StepCentered())))

    # η is averaged over an interval on the coarse grid
    for ic ∈ 1:nc
        # ηc_{ic} = 1/m ∑_{i ∈ (m(ic-1)+1):(m(ic)) }(η_i)     ic = 1, 2, ..., Nt÷m
        gridc.η[ic] = @views dilate_inv(sum(dilate, gridf.η[m*(ic-1)+1:m*ic])/m)
    end
    gridc.ηr .= gridc.η

    # compute τ correction
    # τ_ic    = rf_{m(ic)} - rc_ic
    if !skip
        rf = gridc.scoarsen(selectdim(residual(gridf), D, c_points(gridf, StepCentered())))
        rc = residual(gridc)

        gridc.τ .+= rf .- rc
    else
        fill!(gridc.τ, zero(SC))
    end

    return 
end

function prolongation!(gridf::TimeGrid{T, SF, D}, gridc::TimeGrid{T, SC, D}; interp=true) where {T, SF, SC, D}
    nf = num_time_steps(gridf)
    nc = num_time_steps(gridc)
    m = gridf.m

    selectdim(gridf.u, D, c_points(gridf, TimeCentered())) .+= gridc.srefine(gridc.u .- gridc.r)
    gridc.r .= gridc.u

    # coarse grid correction for η
    for ic ∈ 1:nc
        # ηc_{ic} = 1/m ∑_{i ∈ (m(ic-1)+1):(m(ic)) }(η_i)     ic = 1, 2, ..., Nt÷m
        @views gridf.η[m*(ic-1)+1:m*ic] .= dilate_inv.(clamp.(dilate.(gridf.η[m*(ic-1)+1:m*ic]) .+ dilate(gridc.η[ic]) .- dilate(gridc.ηr[ic]), 0., 2.))
    end
    gridc.ηr .= gridc.η
    if interp
        f_relax!(gridf)
    end
    return
end

# like findfirst, but for sorted data such that f.(x) = [0, 0, ..., 0, 1, 1, ..., 1]
# so I can use bisection to narrow down on the crossover point
function findfirst_bisect(f, x, guess=1)
    a = 1
    b = lastindex(x)

    # check trivial cases
    f(x[a]) && return a
    f(x[b]) || return nothing

    if guess != 0
        i = guess
    else
        i = b ÷ 2
    end

    # TODO: if we assume the crossover point is nearby the guess, we should really do an inverse bisection,
    # checking point i, then i+1, then i+2, then i+4, until we find a small interval we can use bisection on
    itruth = f(x[i])

    # narrow down crossover point with reverse bisection, assuming the guess is close
    δ = itruth ? -1 : 1
    while !(itruth ⊻ f(x[i+δ]))
        δ *= 2
    end

    a, b = sort((i+δ÷2, i+δ))

    # now use bisection if the interval is still too big
    while b - a > 2
        if f(x[i])
            b = i
        else
            a = i
        end
        i = a + (b - a)÷2
    end

    return @views a + findfirst(f, x[a:b]) - 1
end

function closest_prior_timepoint(ti, td, guess)
    # id = findfirst(>=(ti), td)
    id = findfirst_bisect(>=(ti), td, guess+1)
    if isnothing(id)
        return lastindex(td)
    else
        return id - 1
    end
end

function regrid!(u::AbstractVector, ud, t, td, sys::AbstractDiffEQ, disc::AbstractTimeDiscretization)
    Nt = num_time_steps(u)

    u[1] = ud[1]
    @threads for i ∈ 1:Nt
        id = closest_prior_timepoint(t[i+1], td, i)
        u[i+1] = f_step(ud[min(id+1, Nt+1)], ud[id], 0., t[i+1] - td[id], disc, sys)
    end
    return u
end

function regrid!(u, ud, t, td, sys::AbstractDiffEQ, disc::AbstractTimeDiscretization)
    Nt = num_time_steps(u)

    time_slice(u, 1) .= time_slice(ud, 1)
    @threads for i ∈ 1:Nt
        id = closest_prior_timepoint(t[i+1], td)
        time_slice(u, i+1) .= time_slice(ud, min(id+1, Nt+1))
        time_slice(u, i+1) .= f_step(time_slice(u, i+1), time_slice(ud, id), 0., t[i+1] - td[id], disc, sys)
    end
    return u
end

function regrid!(grid::TimeGrid{T, S, D}) where {T, S, D}
    td = [0.; cumsum(dilate.(grid.η, grid.Δt))]
    t = grid.Δt .* (0:grid.Nt)
    dil_mean = td[end] / t[end]

    t *= dil_mean

    ud = copy(grid.u)

    regrid!(grid.u, ud, t, td, grid.sys, grid.disc)

    fill!(grid.η, dilate_inv(dil_mean))

    return 
end

function V_cycle!(grids; lsr_fine=true, lsr_coarse=true, kwargs...) 
    # solve or relax on the coarse grid
    nl = length(grids)
    if nl == 1
        if lsr_coarse
            lsr!(grids[1]; kwargs...)
        else
            solve!(grids[1])
        end
        return
    end

    # pre smoothing
    if lsr_fine
        lsr!(grids[1]; kwargs...)
    else
        f_relax!(grids[1])
    end

    # coarse grid correction
    restrict!(grids[1], grids[2])
    V_cycle!(grids[2:end]; lsr_fine=lsr_fine, lsr_coarse=lsr_coarse, kwargs...)
    prolongation!(grids[1], grids[2])

    # post smoothing
    if lsr_fine
        lsr!(grids[1]; kwargs...)
    else
        f_relax!(grids[1])
    end
    return
end

function skip_down_cycle!(grids) 
    nl = length(grids)
    if nl > 1
        restrict!(grids[1], grids[2]; skip=true)
        skip_down_cycle!(grids[2:end])
    end
    return
end

function down_cycle!(grids, relax::Function; relax_kwargs...)
    if length(grids) > 1
        relax(grids[1]; relax_kwargs...)
        restrict!(grids[1], grids[2])
        down_cycle!(grids[2:end], relax; relax_kwargs...)
    end
    return
end

function down_cycle!(grids, relax::Function, per_lvl_args) 
    if length(grids) > 1
        relax(grids[1]; per_lvl_args[1]...)
        restrict!(grids[1], grids[2])
        down_cycle!(grids[2:end], relax, per_lvl_args[2:end])
    end
    return
end

function up_cycle!(grids, relax::Function; regrid=false, relax_kwargs...) 
    nl = length(grids)
    if nl > 1
        up_cycle!(grids[2:end], relax; relax_kwargs...)
        prolongation!(grids[1], grids[2])
        relax(grids[1]; relax_kwargs...)
        if regrid
            regrid!(grids[1])
        end
    end
    return
end

function up_cycle!(grids, relax::Function, per_lvl_args)
    nl = length(grids)
    if nl > 1
        up_cycle!(grids[2:end], relax, per_lvl_args[2:end])
        prolongation!(grids[1], grids[2])
        relax(grids[1]; per_lvl_args[1]...)
    end
    return
end

function V_cycle!(grids, relax::Function; relax_kwargs...) 
    down_cycle!(grids, relax; relax_kwargs...)
    relax(grids[end]; relax_kwargs...)
    up_cycle!(grids, relax; relax_kwargs...)
end

function V_cycle!(grids, relax::Function, per_lvl_args) 
    down_cycle!(grids, relax, per_lvl_args)
    relax(grids[end]; per_lvl_args[end]...)
    up_cycle!(grids, relax, per_lvl_args)
end

function F_up!(grids, relax::Function, num_V::Integer=1; relax_kwargs...) 
    nl = length(grids)
    if nl > 1
        F_up!(grids[2:end], relax, num_V; relax_kwargs...)
        prolongation!(grids[1], grids[2])
        for _ = 1:num_V
            V_cycle!(grids, relax; relax_kwargs...)
        end
    end
    return
end

function F_up!(grids, relax::Function, per_lvl_args, num_V::Integer=1)
    nl = length(grids)
    if nl > 1
        F_up!(grids[2:end], relax, per_lvl_args[2:end], num_V)
        prolongation!(grids[1], grids[2])
        for _ = 1:num_V
            V_cycle!(grids, relax, per_lvl_args)
        end
    end
    return
end

function F_cycle!(grids, relax::Function; num_V=1, relax_kwargs...)
    down_cycle!(grids, relax; relax_kwargs...)
    relax(grids[end]; relax_kwargs...)
    F_up!(grids, relax, num_V; relax_kwargs...)
    return
end

function F_cycle!(grids, relax::Function, per_lvl_args; num_V=1)
    down_cycle!(grids, relax, per_lvl_args)
    relax(grids[end]; per_lvl_args[end]...)
    F_up!(grids, relax, per_lvl_args, num_V)
    return
end
