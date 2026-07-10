using DifferentialEquations
using Random
include("../../models/bsm1_classic.jl")
function gen_bsm1_noised_data(noise_var, rng::MersenneTwister; ndata=10)
	prob = bsm1_infl("src/data/influent/influent_data.txt")
	res = solve(prob, Rodas4())
	time = LinRange(prob.tspan..., ndata)
	data = [res(t) .+ sqrt(noise_var)*randn(rng, length(prob.u0)) for t in time]
	return data, time
end

