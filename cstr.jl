using DifferentialEquations

function cstr_model()
## Model CSTR
D(t) = 0.35
ωₜ = 1 / 50;
sf = 1.0
Y = 0.4
kₛ = 0.1
μₘ = 0.5

Dₜ(t) = D(t) * exp.(1.0 * sin.(2π * ωₜ * t));
sfₜ(t) = exp.(0.0 * sin.(2π * ωₜ * t));
μ(s, μₘ, kₛ) = μₘ * s / (kₛ + s)


p = (μₘ, kₛ, Y, Dₜ)
u₀ = [4.0; 0.0]

function monod(du, u, p, t)
  du[1] = μ(u[2], p[1], p[2]) * u[1] - p[4](t) * u[1]
  du[2] = p[4](t) * (sf - u[2]) - μ(u[2], p[1], p[2]) / p[3] * u[1]
  nothing
end

function monod(u, p, t)
  return [
    μ(u[2], p[1], p[2]) * u[1] - p[4](t) * u[1],
    p[4](t) * (sf - u[2]) - μ(u[2], p[1], p[2]) / p[3] * u[1]
  ]
end

tspan = (0.0, 200.0)
prob_CSTR = ODEProblem(monod, u₀, tspan, p)

return prob_CSTR
end
