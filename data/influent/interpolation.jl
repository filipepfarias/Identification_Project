
function simple_interpolation(x, f)
	n = length(x)
	function inter(t)
		res = 0.0
		for i in 2:n
			coef = if_else(t > x[i-1], 1.0, 0.0)*if_else(t < x[i], 1.0, 0.0)
			res += coef*((b-a)*(t-x[i-1])+a)
		end
	end
end
