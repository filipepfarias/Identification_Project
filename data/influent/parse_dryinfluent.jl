
data = readlines("src/data/influent/dryinfluent.ascii")
data = hcat([parse.(Float64, split(d, "\t")[1:10]) for d in data]...)

time = data[1,:]
ze = zeros(length(time))
salk = 7.0 .* ones(length(time))
# correct order
ndata = hcat(data[1,:],
	 data[2,:],
	 data[3,:],
	 data[4,:],
	 data[5,:],
	 ze,
	 ze,
	 ze,
	 ze,
	 data[6,:],
	 data[7,:],
	 data[8,:],
	 salk,
	 data[9,:],
	 time)
ndata = [join(col, " ") for col in eachcol(ndata)]
ndata = join(ndata,"\n")
open("src/data/influent/dryinfluent.txt","w") do f
	write(f,ndata)
end

