function alpha = ArmijoLineSearch(objHandle,objVal,x,dx,dir, options)
% ARMIJOLINESEARCH runs a simple backtracking line search with armijo
% conditions
% INPUT
%   objHandle : function handle to the objective function
%   objVal : objective function value in x
%   x : current point 
%   dx : current gradient in x
%   dir : search direction in x, usually -dx


if nargin == 5
    options = struct;
end

if ~isfield(options, 'alpha_start')
    options.alpha_start = 1;
end

if ~isfield(options, 'reduction_factor')
    options.reduction_factor = 1/2;
end

if ~isfield(options, 'c')
    options.c = 1e-4;
end


alpha        = options.alpha_start;

while objHandle(x+alpha*dir) > objVal + options.c*alpha*dir'*dx

    alpha = options.reduction_factor*alpha;
    
    if alpha < 10*eps
        warning("alpha doesn't fit armijo")
        alpha = eps;
        return
    end
    
end

end