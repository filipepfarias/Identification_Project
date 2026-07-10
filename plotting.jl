using GLMakie

sol_CSTR = solve(cstr_model(), RK4());

f = Figure()
ax1 = Axis(f[1,1])
lines!(ax1, sol_CSTR.t, t -> sol_CSTR(t)[1],
    label="Original state 1")
lines!(ax1, sol_CSTR.t, t -> sol_CSTR(t)[2],
    label="Original state 2")

scatter!(ax1,t_sim,getindex.(data,[1]), label="Data state 1",
    color = :transparent, strokecolor=Makie.wong_colors()[1], strokewidth = 1,
    )
scatter!(ax1,t_sim,getindex.(data,[2]), label="Data state 2",
    color = :transparent, strokecolor=Makie.wong_colors()[2], strokewidth = 1,
    )

lines!(ax1,t,x_opt[:,1], 
    label="Optimized state 1",
    linestyle=:dash
    )
lines!(ax1,t,x_opt[:,2], 
    label="Optimized state 2",
    linestyle=:dash
    )
axislegend(ax1,nbanks=2)

ax2 = Axis(f[2,1],
    limits = (nothing, nothing, 0, 5))
lines!(ax2, sol_CSTR.t, cstr_model().p[4](sol_CSTR.t),
    label="Original Input")
lines!(ax2, t, u_opt, 
    label="Fitted Input")
axislegend(ax2)

Label(f[3,1],
    tellwidth = false,
     "Original/Fitted parameters: $(sol_CSTR.prob.p[1])/$(round(p_opt[1],digits=3)), $(sol_CSTR.prob.p[2])/$(round(p_opt[2],digits=3)), $(sol_CSTR.prob.p[3])/$(round(p_opt[3],digits=3))", font=:bold)

f
# plot(p1,p2, layout=(2,1))