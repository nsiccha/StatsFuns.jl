@testitem "F and Student-t extreme tails" begin
    using StatsFuns

    # https://github.com/JuliaStats/StatsFuns.jl/issues/239
    # Independent closed forms when one F degree of freedom is two.
    # Check both tails and their logarithms, including logarithms close to zero.
    @testset "F closed forms ($T)" for T in (Float32, Float64)
        for ν in T.((0.2, 1, 2, 10)), x in T.((0.01, 1, 1.0e6, 1.0e18))
            logp, logq = setprecision(256) do
                n, z = BigFloat(ν), BigFloat(x)
                # F(ν, 2): cdf = (1 + 2 / (ν * x))^(-ν / 2)
                lp = -n / 2 * log1p(2 / (n * z))
                # F(2, ν): ccdf = (1 + 2 * x / ν)^(-ν / 2)
                lq = -n / 2 * log1p(2 * z / n)
                (lp, lq)
            end
            for (ν1, ν2, p, q, lp, lq) in (
                    (ν, T(2), exp(logp), -expm1(logp), logp, log(-expm1(logp))),
                    (T(2), ν, -expm1(logq), exp(logq), log1p(-exp(logq)), logq),
                )
                for (f, expected) in (
                        (fdistcdf, p), (fdistccdf, q),
                        (fdistlogcdf, lp), (fdistlogccdf, lq),
                    )
                    actual = @inferred f(ν1, ν2, x)
                    @test actual isa T
                    @test actual ≈ T(expected) rtol = max(64eps(T), 1.0e-13)
                end
            end
        end
    end

    @testset "Cauchy tails ($T)" for T in (Float32, Float64)
        for x in T.((100, 1.0e6, 1.0e9))
            p = setprecision(256) do
                atan(inv(BigFloat(x))) / big(π)
            end
            @test tdistcdf(T(1), -x) ≈ T(p) rtol = 64eps(T)
            @test tdistccdf(T(1), x) ≈ T(p) rtol = 64eps(T)
            @test tdistlogcdf(T(1), -x) ≈ T(log(p)) rtol = 64eps(T)
            @test tdistlogccdf(T(1), x) ≈ T(log(p)) rtol = 64eps(T)
            @test tdistlogcdf(T(1), x) ≈ T(log1p(-p)) rtol = 64eps(T)
            @test tdistlogccdf(T(1), -x) ≈ T(log1p(-p)) rtol = 64eps(T)
        end
    end

    # 100-digit incomplete-beta reference for the original fractional-df case.
    @test tdistcdf(0.2, -1.0e6) ≈ 0.02371964367753373453132615287588497 rtol = 1.0e-13
end
