# functions related to F distribution

# Julia implementations
fdistpdf(ν1::Real, ν2::Real, x::Real) = exp(fdistlogpdf(ν1, ν2, x))

fdistlogpdf(ν1::Real, ν2::Real, x::Real) = fdistlogpdf(promote(ν1, ν2, x)...)
function fdistlogpdf(ν1::T, ν2::T, x::T) where {T <: Real}
    a = ν1 / 2
    b = ν2 / 2
    lbeta = logbeta(a, b)
    # in terms of the beta variate `u = ν1 * x / (ν1 * x + ν2)` the density is
    # `u^a * (1 - u)^b / (x * beta(a, b))`; `r` below is `u / (1 - u)`, formed so that a large
    # `x` does not overflow it
    ν1ν2 = ν1 / ν2
    r = ν1ν2 * x
    invr = inv(r)
    return if x < 0
        # outside of the support, where `log(x)` would error as well
        oftype(lbeta, -Inf)
    elseif isfinite(invr)
        # `log(u) = -log1p(1 / r)` and `log(1 - u) = -log1p(r)`, symmetric in the two degrees
        # of freedom, so no two terms growing like `ν1 * log(ν1)` cancel. `r` overflows for the
        # largest `x`, where `log1p(r)` is ordinary and both summands below are positive
        blog1pr = isinf(r) ? xlogy(b, ν1ν2) + xlogy(b, x) : xlog1py(b, r)
        -xlog1py(a, invr) - blog1pr - log(x) - lbeta
    else
        # `r` is zero or subnormal: `x == 0`, `r` underflowing, or `NaN`. The textbook form
        # covers all three - `log1p(r)` keeps a tiny `r` that `log(1 + r)` would absorb, and
        # splitting `log(u)` into `log(ν1 / ν2) + log(x)` needs no precision from `r`
        xlogy(a, ν1ν2) + xlogy(a - 1, x) - xlog1py(a + b, r) - lbeta
    end
end

for (f, fc) in (("cdf", "ccdf"), ("ccdf", "cdf"), ("logcdf", "logccdf"), ("logccdf", "logcdf"))
    ff = Symbol("fdist" * f)
    bf = Symbol("beta" * f)
    bfc = Symbol("beta" * fc)
    @eval function $ff(ν1::T, ν2::T, x::T) where {T <: Real}
        # Evaluate the smaller beta argument directly, swapping the parameters and tail
        # when r > 1. Forming u close to one loses the complementary tail's precision.
        r = (ν1 * max(0, x)) / ν2
        if r > 1
            return $bfc(ν2 / 2, ν1 / 2, inv(1 + r))
        end
        u = r / (1 + r)
        return $bf(ν1 / 2, ν2 / 2, u)
    end
    @eval $ff(ν1::Real, ν2::Real, x::Real) = $ff(promote(ν1, ν2, x)...)
end

# The quantile is `ν2 * x / (ν1 * (1 - x))` for the quantile `x` of `u ~ Beta(ν1 / 2, ν2 / 2)`,
# with `1 - x` computed directly (inaccurate for skewed `u`: JuliaMath/SpecialFunctions.jl#578)
function fdistinvcdf(ν1::T, ν2::T, p::T) where {T <: Real}
    x, xc = _beta_inc_inv(ν1 / 2, ν2 / 2, p)
    return (ν2 * x) / (ν1 * xc)
end
function fdistinvccdf(ν1::T, ν2::T, q::T) where {T <: Real}
    xc, x = _beta_inc_inv(ν2 / 2, ν1 / 2, q)
    return (ν2 * x) / (ν1 * xc)
end

# `betainvlogcdf` returns only `x`, so we solve for whichever of `u` and `1 - u ~ Beta(ν2 / 2, ν1 / 2)`
# is at most 1/2 at the quantile, which is `u` if and only if `logp ≤ log(P(u ≤ 1/2))`
function fdistinvlogcdf(ν1::T, ν2::T, logp::T) where {T <: Real}
    a = ν1 / 2
    b = ν2 / 2
    if logp ≤ log(first(beta_inc(a, b, one(a) / 2)))
        x = betainvlogcdf(a, b, logp)
        return (ν2 * x) / (ν1 * (1 - x))
    else
        xc = betainvlogccdf(b, a, logp)
        return (ν2 * (1 - xc)) / (ν1 * xc)
    end
end
# `1 / F ~ F(ν2, ν1)`
fdistinvlogccdf(ν1::T, ν2::T, logq::T) where {T <: Real} = inv(fdistinvlogcdf(ν2, ν1, logq))

for f in ("invcdf", "invccdf", "invlogcdf", "invlogccdf")
    ff = Symbol("fdist" * f)
    @eval $ff(ν1::Real, ν2::Real, y::Real) = $ff(promote(ν1, ν2, y)...)
end
