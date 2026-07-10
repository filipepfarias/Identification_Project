function gen_influent_data()
	total_time = 15.0 # days
	dt = 15.0 / 24.0 / 60.0 # min
	time = [0.0:dt:total_time;]
	omega = 2*pi / 3
	wave = [ 1 + 0.1sin(omega*t) for t in time]
	
	SI_AIN = 3.0000000e+01  .* wave
	SS_AIN = 6.9500000e+01  .* wave
	XI_AIN = 5.1200000e+01  .* wave
	XS_AIN = 2.0232000e+02  .* wave
	XBH_AIN = 2.8170000e+01 .* wave
	XBA_AIN = 0.0           .* wave
	XP_AIN = 0.0            .* wave
	SO_AIN = 0.0            .* wave
	SNO_AIN = 0.0           .* wave
	SNH_AIN = 3.1560000e+01 .* wave
	SND_AIN = 6.9500000e+00 .* wave
	XND_AIN = 1.0590000e+01 .* wave
	SALK_AIN = 7.0          .* wave

	Q_IN = 18446.0          .* wave
	
	data = hcat(SI_AIN,
		SS_AIN,
		XI_AIN,
		XS_AIN,
		XBH_AIN,
		XBA_AIN,
		XP_AIN,
		SO_AIN,
		SNO_AIN,
		SNH_AIN,
		SND_AIN,
		XND_AIN,
		SALK_AIN,
		Q_IN,
		time)
	data = [join(row, " ") for row in eachcol(data)]
	data = join(data, "\n")
	open("src/data/influent/influent_data.txt", "w") do f
		write(f, data)
	end
end

gen_influent_data()
