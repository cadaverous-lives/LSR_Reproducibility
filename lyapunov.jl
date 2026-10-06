using LinearAlgebra
include("methods.jl")

function rcols_I(N, rank, T)
    out = zero(MMatrix{N, rank, T})
    for j ∈ 1:rank
        out[j, j] = one(T)
    end
    SMatrix(out)
end

lyap_vectors(u, h::Number, step, sys; kwargs...) = lyap_vectors(u, fill(h, size(u)[end]-1), step, sys; kwargs...)

function lyap_vectors(u::AbstractVector{V}, h::AbstractVector, step, sys; rank=typemax(Int), pad=0) where V <: StaticVector{N, T} where {N, T}
    nt = length(u) - 1
    tfinal = sum(h)
    tpad = sum(h[1:pad])

    rank = min(rank, N)

    Ψ⁻ = zeros(SMatrix{N, rank}, nt+1)
    Ψ⁺ = zeros(SMatrix{N, rank}, nt+1)
    Γ  = zeros(SMatrix{N, rank}, nt+1)
    Θ  = zeros(SMatrix{N, rank}, nt+1)
    # Ψ⁻[1] = SMatrix(qr(randn(SMatrix{N, rank, T})).Q)
    # Ψ⁺[end] = SMatrix(qr(randn(SMatrix{N, rank, T})).Q)
    Ψ⁻[1] = rcols_I(N, rank, T)
    Ψ⁺[end] = rcols_I(N, rank, T)

    λ = zero(MVector{rank, real(T)})

    # forward and backward vectors
    for i = 1:nt
        Ψ⁻[i+1] = step.Φ_u(u[i], h[i], sys) * Ψ⁻[i]
        F = qr(Ψ⁻[i+1])
        Ψ⁻[i+1] = Matrix(F.Q)*SDiagonal{rank, T}(sign.(diag(F.R)))
    end

    for i = nt:-1:1
        Ψ⁺[i] = step.Φ_u(u[i], h[i], sys)' * Ψ⁺[i+1]
        F = qr(Ψ⁺[i])
        Ψ⁺[i] = SMatrix(F.Q) * SDiagonal{rank, T}(sign.(diag(F.R)))
    end

    # covariant vectors
    @views for i = 1:nt+1
        A⁺, A⁻ = lu(Ψ⁺[i]' * Ψ⁻[i], NoPivot())
        Γ[i] = Ψ⁺[i] * A⁺

        B⁻, B⁺ = lu(Ψ⁻[i]' * Ψ⁺[i], NoPivot())
        Θ[i] = Ψ⁻[i] * B⁻

        # normalize
        Γ[i] = Γ[i]*SDiagonal{rank, T}(1 ./ norm.(eachcol(Γ[i])))
        Θ[i] = Θ[i]*SDiagonal{rank, T}(1 ./ norm.(eachcol(Θ[i])))

        i-1 == 0 && continue
        # lyapunov exponents
        F = step.Φ_u(u[i-1], h[i-1], sys)
        σ = diag(Γ[i] \ (F * Γ[i-1]))
        Γ[i] = Γ[i]*SDiagonal{rank, T}(sign.(σ))
        if (i > pad)
            λ .+= (log.(abs.(σ))/h[i-1])*h[i-1]/(tfinal-tpad)
        end
    end

    println("Lyapunov exponents λ = $λ")

    return (Ψ⁻[pad+1:end-pad], Ψ⁺[pad+1:end-pad], Γ[pad+1:end-pad], Θ[pad+1:end-pad])
end

function lyap_vectors(u::AbstractMatrix{T}, h::AbstractVector, step, sys; rank=typemax(Int), pad=0) where T
    nx, nt = (size(u, 1), size(u, 2)-1) 
    tfinal = sum(h)
    tpad = sum(h[1:pad])

    rank = min(rank, nx)

    Ψ⁻ = zeros(T, nx, rank, nt+1)
    Ψ⁺ = zeros(T, nx, rank, nt+1)
    Γ  = zeros(T, nx, rank, nt+1)
    Θ  = zeros(T, nx, rank, nt+1)
    Ψ⁻[:, :, 1] .= qr!(randn(T, nx, rank)).Q.factors
    Ψ⁺[:, :, end] .= qr!(randn(T, nx, rank)).Q.factors

    λ = zeros(T, rank)

    # forward and backward vectors
    @views for i = 1:nt
        Ψ⁻[:, :, i+1] .= step.Φ_u(u[:, i], h[i], sys) * Ψ⁻[:, :, i]
        F .= qr(Ψ⁻[:, :, i+1])
        Ψ⁻[:, :, i+1] .= F.Q * Diagonal(sign.(diag(F.R)))
    end

    @views for i = nt:-1:1
        Ψ⁺[:, :, i] .= step.Φ_u(u[:, i], h[i], sys)' * Ψ⁺[:, :, i+1]
        F .= qr(Ψ⁺[:, :, i])
        Ψ⁺[:, :, i] .= F.Q * Diagonal(sign.(diag(F.R)))
    end

    # covariant vectors
    @views for i = 1:nt+1
        A⁺, A⁻ .= lu!(Ψ⁺[:, :, i]' * Ψ⁻[:, :, i], NoPivot())
        Γ[:, :, i] .= Ψ⁺[:, :, i] * A⁺

        B⁻, B⁺ .= lu!(Ψ⁻[:, :, i]' * Ψ⁺[:, :, i], NoPivot())
        Θ[:, :, i] .= Ψ⁻[:, :, i] * B⁻

        # normalize
        Γ[:, :, i] .= Γ[:, :, i]*Diagonal(1 ./ norm.(eachcol(Γ[:, :, i])))
        Θ[:, :, i] .= Θ[:, :, i]*Diagonal(1 ./ norm.(eachcol(Θ[:, :, i])))

        i-1 == 0 && continue

        # lyapunov exponents
        F = step.Φ_u(u[:, i-1], h[i-1], sys)
        σ = diag(Γ[:, :, i] \ (F * Γ[:, :, i-1]))
        Γ[:, :, i] = Γ[:, :, i]*Diagonal(sign.(σ))
        if (i > pad)
            λ .+= log.(abs.(σ)) ./ (tfinal-tpad)
        end
    end

    println("Lyapunov exponents λ = $λ")

    return (Ψ⁻[:, :, pad+1:end-pad], Ψ⁺[:, :, pad+1:end-pad], Γ[:, :, pad+1:end-pad], Θ[:, :, pad+1:end-pad])
end

function project_onbasis(u, Γ, orthonormal=true)
    if orthonormal
        return Γ' * u
    else
        # \ automatically uses the psuedoinverse in case Γ is rectangular
        return Γ \ u
    end
end
