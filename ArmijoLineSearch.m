function alpha = ArmijoLineSearch(objHandle,objVal,x,dx,dir, options)
% ARMIJOLINESEARCH runs a simple backtracking line search with Armijo
% condition
% INPUT
%   objHandle : function handle to the objective function
%   objVal : objective function value in x
%   x : current point 
%   dx : current gradient in x
%   dir : search direction in x, usually -dx
%   options : optional input struct to change the default parameters of the line search
%           - options.alpha_start: starting step size (default: 1)
%           - options.reduction_factor: factor with which alpha will be multiplied in each iteration when the Armijo condition is not yet met (default: 1/2)
%           - options.c: the control parameter of the Armijo condition, c in ]0, 1[ (default: 1e-4)


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
