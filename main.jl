include("cstr.jl");

using Distributions, JuMP, Ipopt, DataInterpolations

function generate_noisy_data()
    # Choose a heavy-tailed noisy distribution
    d = TDist(1.5)
    nd = 100

    sol_CSTR = solve(cstr_model(),RK4());

    t_sim = range(cstr_model().tspan...,nd)

    t_sim, noisy_sim = collect(t_sim), [sol_CSTR(t) + .2abs.(rand(d,2)) for t in t_sim]

    return t_sim, noisy_sim
end

function solve_constrained_identification()
    prob_CSTR = cstr_model();
    f = prob_CSTR.f.f
    p = prob_CSTR.p
    t_sim, noisy_sim = generate_noisy_data()
    dt = diff(t_sim)
    n = 800
    nd = length(noisy_sim)

    model = Model(Ipopt.Optimizer)

    @variable(model, u[1:n],lower_bound = 0.0)
    @variable(model, x[1:n,1:2],lower_bound = 0.0)
    @variable(model, residuals[1:nd])
    @variable(model, lambda)

    # Collocation constraint using Chebyshev points
    # t = .5 * (1 .- cos.( (0:n-1) ./(n-1) .* pi )) .* diff([prob_CSTR.tspan...]) .+ prob_CSTR.tspan[1]
    t = range(prob_CSTR.tspan...,n)
    dt = diff(t)
    for i in 1:n-1
        
        ## # Uncomment to use RK4
        ## k1 = f(x[i,:],(p[1:3]..., t->u[i]),t[i]+dt[i])
        ## k2 = f(x[i,:] + dt[i]/2 * k1,(p[1:3]..., t->u[i]),t[i]+dt[i]/2)     
        ## k3 = f(x[i,:] + dt[i]/2 * k2,(p[1:3]..., t->u[i]),t[i]+dt[i]/2)    
        ## k4 = f(x[i,:] + dt[i] * k3,(p[1:3]..., t->u[i]),t[i]+dt[i])        
        
        ## @constraint(model, 
        ##     - x[i,:] + x[i+1,:] + dt[i]/6 * (k1 + 2*k2 + 2*k3 + k4) .== 0
        ## )
        
        # Uncomment to use Euler
        k = f(x[i,:],(p[1:3]..., t->u[i]),t[i])
        @constraint(model, 
            - x[i,:] + x[i+1,:] + dt[i] * k .== 0
        )
    end

    for i in 1:nd
        # Find bracketing index in collocation grid
        idx = findfirst(t .>= t_sim[i])
        
        if idx == 1
            # Use first point directly
            @constraint(model, 
                residuals[i] == (x[1,1] - noisy_sim[i][1])^2 + (x[1,2] - noisy_sim[i][2])^2
            )
        elseif idx === nothing || idx > n
            # Use last point directly
            @constraint(model, 
                residuals[i] == (x[n,1] - noisy_sim[i][1])^2 + (x[n,2] - noisy_sim[i][2])^2
            )
        else
            # Linear interpolation
            a = (t_sim[i] - t[idx-1]) / (t[idx] - t[idx-1])
            
            @constraint(model, 
                residuals[i] == ((1-a)*x[idx-1,1] +a*x[idx,1] - noisy_sim[i][1])^2 + 
                                ((1-a)*x[idx-1,2] +a*x[idx,2] - noisy_sim[i][2])^2
            )
        end
    end
    @constraint(model, 
            residuals[1] == (x[1,1] - noisy_sim[1][1])^2 + (x[1,2] - noisy_sim[1][2])^2
        )
    @constraint(model, 
            residuals[nd] == (x[n,1] - noisy_sim[nd][1])^2 + (x[n,2] - noisy_sim[nd][2])^2
        )

    @objective(model, Min, sum(residuals) + .01sum(u[i]^2 for i in 1:n))
    
    optimize!(model)
    
    try
        display("Lambda: "*string(value(lambda)))
    catch e 
        nothing
    end

    return model, t, value.(x), value.(u), t_sim, noisy_sim
end

model, t, x_opt, u_opt, t_sim, data = solve_constrained_identification()

using Plots
closeall()
sol_CSTR = solve(cstr_model(), RK4());
p1 = plot(sol_CSTR,label="Original states")
scatter!(p1,t_sim,hcat(data...)', label="Data")
plot!(p1,t,x_opt, label="Optimized states")

p2 = plot(sol_CSTR.t,cstr_model().p[4](sol_CSTR.t),label="Original Input")
plot!(p2,t,u_opt,label="Optimized Input", ylim=(-.5,5))

plot(p1,p2, layout=(2,1))
