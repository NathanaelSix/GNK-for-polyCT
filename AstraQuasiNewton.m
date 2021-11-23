classdef AstraQuasiNewton < handle
    %ASTRAQUASINEWTON abstract class implementing general quasi newton methods
    %   ASTRAQUASINEWTON implements all necessary and expected parts
    %   for quasi newton methods, every specific implementation of
    %   objective class needs to be implemented in a derived class. 
    %   We specifically expect these classes
    %   to require some forward projector operator A from astra, and
    %   minimize and objective function of the form 
    %   1/2||M_A(x) - b||_2^2 + R(x) with M_A some projection model
    %   Main reconstruction method is Gauss_Newton-Krylov, however,
    %   also includes the option to run Gradient descent and L-BFGS on the
    %   same objective function
    % Date: 22-11-2021
    % Author: Nathanael Six
    % Affiliation: imec-VisionLab, University of Antwerp, Belgium
    % Contact: nathanael.six@uantwerpen.be
    % License: GPLv3
    
properties
    A   = [] % projection operator object, needs most of the opTomo functionality
    b   = [] % projection data
    x   = [] % current guess of the solution
    N   = [] % number of variables in the system
    fx  = [] % current result of M_A(self.x)
    gradx=[] % current gradient of the objective function
    objx = [] % current result of 1/2||M_A(x) - b||_2^2 + R(x)
    hist = struct % struct for histories
    gt  = [] % ground truth image if available
    method_counter = struct % struct to track how many times each method is called
    
    % options
    calculate_errors = true % if false, no history of errors is kept
    solver = 'minres'
    solver_handle = []
    restart_gmres = 10 % value only used when solver = 'gmres'
    iters_newton = 20 % default value, expected to be changed by user
    iters_minres = 50 % default value, expected to be changed by user
    threshold_minres = 10^-6 % default value, expected to be changed by user
    tolX = 10^-8 % default value, expected to be changed by user
    tolFun = 10^-8 % default value, expected to be changed by user
    verbose = true % default value, expected to be changed by user
    tol_NK = 10^-4 % default value, expected to be changed by user
    c = 10^-4 % default value, expected to be changed by user
    
    % gradient descent options
    grad_desc_step_size = 1e-2  % in case of fixed step length
    grad_desc_line_search = 'bb'; % 'none' for fixed step length, 
                                  % 'backtracking' for armijo line search,
                                  % 'bb' for Barzilai-Borwein steps
    % stored for gradient methods
    xp = []
    gradp = []
    objp = []
    s_dirp = []
                                  
    
    %flags
    is_initialized = false 
    
    % optional regularization 
    is_regularized = false
    reg_lambda = 0
    reg_method = []
    tv_matrix = []
    tv_beta = 1e-3; % default value, expected to be changed by user
    
    % lbfgs only
    bfgs_line_search = 'none' % 'none', 'backtracking'
    bfgs_iters = 100
    bfgs_mem = 5   % default value, expected to be changed by user
    bfgs_tol = 1e-8 % default value, expected to be changed by user
    
    
end %properties

methods (Abstract)
    out  = f(self, args)
    Jy   = jacobian(self, y, is_transposed)
end % methods


methods
    %% construction and initialization
    function self = AstraQuasiNewton(A, x0, b, options)
        %ASTRAQUASINEWTON Construct an instance of this class
        %   inputs:
        %   - A     :opTomo operator
        %   - x0    :initial guess
        %   - b     :projection data
        %   - options:struct with any of the options properties
        %       options will also fill in any mentioned properties of
        %       superclasses!
        narginchk(3,4);
        self.A = A;
        self.x = x0(:);
        self.b = b(:);
        
        self.N = numel(x0);

        if nargin > 3
            if ~isa(options, 'struct')
                error('options field must be a struct')
            end
            prop = properties(self);
            for ii = 1:length(prop)
                % we don't want to allow an option struct to change
                % the hard required values
                if strcmp(prop{ii}, 'A') || strcmp(prop{ii}, 'x') || strcmp(prop{ii}, 'b')
                    continue
                end
                if isfield(options, prop{ii})
                    self.(prop{ii}) = options.(prop{ii});
                end
            end
            
        end
        
        method_names = methods(self);
        for ii = 1:length(method_names)
            self.method_counter.(method_names{ii}) = 0;
        end
        
        % is_regularized, reg_lambda and reg_method can be set with the
        % options struct, or after manually after calling the constructor
        % remember to set ALL when setting manually
        
        % setting default options if not all settings were set
        if numel(self.reg_method) > 0 || self.reg_lambda ~= 0
            self.is_regularized = true;
        end
        
        if self.is_regularized
            if self.reg_lambda == 0
                self.reg_lambda = 1;
                warning('regularization lambda set to default = 1. You can still change these values before running .initialize.')
            end
            if numel(self.reg_method) == 0
                self.reg_method = 'tikhonov';
                warning('regularization method set to default = tikhonov. You can still change these values before running .initialize.')
            end
        end
        

    end%function
    
    function self = initialize(self)
        % initializes the class so as to not do a lot of calculations when
        % one just creates an instance of the class
        % if this step is not ran, self.run() will not work.
        
        self.method_counter.initialize = self.method_counter.initialize + 1;
       
        
         % regularisation initialization
        if strcmpi(self.reg_method, 'tvmin')
            tv_1d = diag(ones(sqrt(self.N), 1)) - diag(ones(sqrt(self.N)-1, 1),1);
            tv_1d = sparse(tv_1d(1:end-1, :));
            
            I = speye(sqrt(self.N));
            self.tv_matrix = [kron(tv_1d, I); kron(I, tv_1d)];
        end
        
        % intiate class variables 
        self.update_x(self.x);
        
        %prepare function handle for chosen solver
        self.solver = lower(self.solver);
        switch self.solver
            case 'minres'
                self.solver_handle = @(H, g, th, it) minres(H, g, th, it);
            case 'gmres'
                self.solver_handle = @(H, g, th, it) gmres(H, g, self.restart_gmres, th, it);
            case 'bicgstab'
                self.solver_handle = @(H, g, th, it) bicgstab(H, g, th, it);
            case 'cgs'
                self.solver_handle = @(H, g, th, it) cgs(H, g, th, it);
            otherwise
                error(['chosen solver name not recognized, options are: ',...
                    'minres, gmres, bicgstab, cgs'])
        end
        
        if self.calculate_errors
            self.hist.obj = [];
            self.hist.data_fid = [];
            self.hist.gt_mse = [];
        end

        self.is_initialized = true;
    end %function
    
    function self = update_x(self, new_x)
        self.method_counter.update_x = self.method_counter.update_x + 1;

        % UPDATE_x updates the class with x as new iterate as current guess
        
        self.x = new_x;
        
        % call all separate update calls 
        % it is assumed that the functions f, grad, and obj can be called
        % without argument to use all needed parts from self." "
        
        self.update_pre_f; % updates subclass specific parts, overload as needed
        self.fx = self.f;
        self.update_pre_grad; % updates subclass specific parts, overload as needed        
        self.gradx = self.grad;       
        self.update_pre_obj; % updates subclass specific parts, overload as needed
        self.objx = self.obj;
        
    end %function
    
    function out = obj(self, z, varargin)
        % important when making a new class deriving from this:
        % the objective function needs to be able to be run on test values
        % during a line search, so running obj(z) has to run independently
        % from stored values, see the nargin check lower. This behavior
        % needs to be reflected in the implented f method as well.
        self.method_counter.obj = self.method_counter.obj + 1;
        if nargin == 1
            z = self.x;
            fz = self.fx;
        else
            fz = self.f(z, varargin{:});
        end
        out = 1/2 * norm(fz - self.b, 2)^2 + self.reg(z);
    end %function
    
    function out = data_error(self, z, varargin)
        % same function as objective, only this one only uses the data
        % fidelity term. Same function in case of no regularisation. Only
        % used for statistics.
        self.method_counter.data_error = self.method_counter.data_error + 1;        
        if nargin == 1
            fz = self.fx;
        else
            fz = self.f(z, varargin{:});
        end
        out = 1/2 * norm(fz - self.b, 2)^2; 
    end %function
    
    function out = mse_gt(self, z)
        self.method_counter.mse_gt = self.method_counter.mse_gt + 1;        
         out = sum((z(:) - self.gt(:)).^2) / self.N;
    end
    
    function out = grad(self, z)
        self.method_counter.grad = self.method_counter.grad + 1;                
        if nargin == 1
            fz = self.fx;
            z = self.x;
        else
            fz = self.f(z);
        end
        rz = fz - self.b;

        out = self.jacobian(rz, true) + self.grad_reg(z);
    end %function
    
     function self = error_statistics(self)
        self.method_counter.error_statistics = self.method_counter.error_statistics + 1;                
        self.hist.obj(end+1) = self.objx;
        self.hist.data_fid(end+1) = self.data_error(self.x);
        if ~isempty(self.gt)
            self.hist.gt_mse(end+1) = self.mse_gt(self.x);
        end
     end % function
        
    %% overloadable functions
    function update_pre_f(~)
    end
    
    function update_pre_grad(~)
    end
    
    function update_pre_obj(~)
    end
    
    function out = reg(self, x)
        self.method_counter.reg = self.method_counter.reg + 1;                
        if ~self.is_regularized
            out = 0;
            return
        end
        
        switch lower(self.reg_method)
            case 'tikhonov'
                out = self.reg_lambda * 1/2 * norm(x, 2)^2;
            case 'tvmin'
                out = self.reg_lambda * sum( sqrt( (self.tv_matrix * x).^2 + self.tv_beta ) );
            otherwise
                error('requested regularisation method not implemented');
        end

    end
    
    function out = grad_reg(self, x)
        self.method_counter.grad_reg = self.method_counter.grad_reg + 1;                
        if ~self.is_regularized
            out = 0;
            return
        end
        switch lower(self.reg_method)
            case 'tikhonov'
                out = self.reg_lambda * x;
            case 'tvmin'
                out = self.reg_lambda * ( self.tv_matrix' * ...
                    ((self.tv_matrix * x) ./ sqrt((self.tv_matrix * x).^2 ...
                    + self.tv_beta)));
            otherwise
                error('requested regularisation method not implemented');
        end
    end
    
    function out = hessian_reg(self, y)
        self.method_counter.hessian_reg = self.method_counter.hessian_reg + 1;                
        if ~self.is_regularized
            out = 0;
            return
        end
        % multiplication of the hessian of the regularisation with y
        switch lower(self.reg_method)
            case 'tikhonov'
                out = self.reg_lambda * y; 
            case 'tvmin'
                out = self.tv_matrix' * (self.tv_beta ./ ...
                    (( (self.tv_matrix*self.x).^2 + self.tv_beta).^(3/2)) ...
                    .* (self.tv_matrix*y));
            otherwise
                error('requested regularisation method not implemented');
        end

    end
    
    %% Hessian calculation
    function Hy = hessian(self, y)
        % method assumes it always needs to operate on properties in self
        % so it's the hessian in self.x
        % we always use the Jacobian approximation for the Hessian of the
        % 1/2||M_A(x) - b||_2^2 part of out objective function
        self.method_counter.hessian = self.method_counter.hessian + 1;   
        
        Hy = self.jacobian(self.jacobian(y, false), true) + ...
                self.hessian_reg(y);
    end %function
    
    
    %% Run Newton Krylov
    function self = run(self)
        % function to run the Gauss Newton Krylov optimization
        % written in part by Jeffrey Cornelis, University of Antwerp
        
     if ~self.is_initialized
        error('Class not intialized')
     end
     start_tic = tic;   
     
     self.hist.nsolves = 0;
     self.hist.minres_its = 0;


     for ii = 1:self.iters_newton
         if self.verbose
             display(['Gauss-Newton-Krylov : iteration ', ...
                 num2str(ii), '/', num2str(self.iters_newton)]);
         end
         Hx = @(y) self.hessian(y);
         [dx,~,RELRES,~,residual] = self.solver_handle(Hx, ...
             -self.gradx, self.threshold_minres, self.iters_minres);
         self.hist.minres_its = self.hist.minres_its + length(residual) - 1;
         self.hist.nsolves = self.hist.nsolves + 1;
         
         if self.calculate_errors
            self.hist.residuals_minres{ii} = residual./residual(1);
         end
         descent_direction = self.gradx'*dx;
         if(descent_direction > 0)
             warning('Step does not produce descent direction')
             return
         end             
         alpha = 1;
         y = self.x + alpha*dx;
         
         obj_test = self.obj(y);
         while (self.objx + self.c*alpha*descent_direction < obj_test)
             alpha = alpha/2;
             if alpha < eps
                 self.hist.updatenorm(ii) = norm(dx);
                 if self.calculate_errors
                     self.hist.fvalnorm(ii) = norm(self.gradx);
                     self.hist.time(ii) = toc(start_tic);
                     self.error_statistics;
                 end
                 disp(['Gauss-Newton-Krylov reached accuracy ',num2str(norm(self.gradx))])
                 warning('Unable to minimize objective function further.')
                 return
             end
             y = self.x + alpha*dx;
             obj_test = self.obj(y);
         end
         
         
         
         %stopping criterion: stop if update is too small
         if self.objx - obj_test < self.tolFun
            self.update_x( y );
            warning('tolFun reached')
            disp(['Gauss-Newton-Krylov reached accuracy ', num2str(norm(self.gradx))])
            self.hist.updatenorm(ii) = norm(dx);
            if self.calculate_errors
                
                self.hist.time(ii) = toc(start_tic);
                self.hist.fvalnorm(ii) = norm(self.gradx);
                self.hist.accuracy_minres(ii) = RELRES;
                self.error_statistics;
            end

            return
         end


         self.update_x( y );
         
         self.hist.updatenorm(ii) = norm(dx);
         if self.calculate_errors
             
             self.hist.time(ii) = toc(start_tic);
             self.hist.fvalnorm(ii) = norm(self.gradx);
             self.hist.accuracy_minres(ii) = RELRES;
             self.error_statistics;
         end
         if norm(self.gradx) < self.tol_NK % convergence check
            disp('Newton-Krylov solver converged to desired accuracy')
            return
         end



         % stopping criterion: stop if update is too small
         if alpha*self.hist.updatenorm(ii) < self.tolX
             warning('tolX reached')
             disp(['Newton-Krylov solver reached accuracy ', ...
                 num2str(norm(self.gradx))])
             return
         end

     end


    end %function 
    
    %% gradient descent
    function self = run_gradient_descent(self, num_grad_iter)
        % function to run gradient descent on the same objective
        % function, for comparison purposes.
        % with fixed stepsize        
        self.method_counter.run_gradient_descent = self.method_counter.run_gradient_descent + 1;

        for ii = 1:(num_grad_iter)
           if strcmpi(self.grad_desc_line_search, 'none')
               alpha = self.grad_desc_step_size / norm(self.gradx, 2);
           elseif strcmpi(self.grad_desc_line_search, 'backtracking')
               alpha = ArmijoLineSearch(@(x) self.obj(x), self.objx, self.x, self.gradx, -self.gradx, struct('c', self.c));
           elseif strcmpi(self.grad_desc_line_search, 'bb')
               if ii == 1
                   alpha = ArmijoLineSearch(@(x) self.obj(x), self.objx, self.x, self.gradx, -self.gradx, struct('c', self.c));                    
               else
                   alpha = abs( dot(self.x-self.xp, self.gradx - self.gradp)) / ...
                   dot(self.gradx - self.gradp, self.gradx - self.gradp);
               end
           else
               error("incorrect gradient descent method chosen. Options are 'none', 'backtracking' or 'bb'")
           end
           self.xp = self.x;
           self.gradp = self.gradx;
           self.update_x(self.xp - alpha*self.gradp);
           if self.calculate_errors
               self.error_statistics;
           end
        end

        
    end %function
    
    function self = run_lbfgs(self)
        self.method_counter.run_lbfgs = self.method_counter.run_lbfgs + 1;
        
        initial_gamma = 1/norm(self.gradx); % very first hessian 
         
        x_k            = NaN*ones(self.N,self.bfgs_mem);
        x_k(:,1)       = self.x;  % starting value for parameters
        objFuncValue = NaN*ones(1,2);
        dx           = NaN*ones(self.N,self.bfgs_mem);

        s_k = ones(self.N,self.bfgs_mem-1) * initial_gamma;
        y_k = ones(self.N,self.bfgs_mem-1);
        r_k = ones(self.bfgs_mem-1,1);
        a_k = ones(1,self.bfgs_mem-1);
        
        % init
        objFuncValue(1) = self.objx ;
        objFuncValue(2) = objFuncValue(1) + 1;
        dx(:,1)          = self.gradx;

        % iterate
        iter      = 0;
        while iter < self.bfgs_iters && abs((objFuncValue(2)-objFuncValue(1))/objFuncValue(1))>self.bfgs_tol && norm(dx(:,1))>self.bfgs_tol
            
            % in the first iteration, H^-1 is approximated as identity
            
            % inverse hessian update
            q = dx(:,1);
            for ii = 1:min(iter,self.bfgs_mem-1)
                a_k(ii) = r_k(ii)*s_k(:,ii)'*q;
                q      = q - a_k(ii)*y_k(:,ii);
            end
            z = s_k(:,1)'*y_k(:,1)/(y_k(:,1)'*y_k(:,1))*q; % this corresponds to H*q where H is approximated as described in Nocedal 9.1
            for ii = min(iter,self.bfgs_mem-1):-1:1
                tmp = r_k(ii)*y_k(:,ii)'*z;
                z = z + s_k(:,ii)*(a_k(ii)-tmp);
            end
            
            % obtain search direction
            dir = -z;            

            % linesearch in search direction for stepsize
            if strcmpi(self.bfgs_line_search, 'none')
                alpha = 1;
            elseif strcmpi(self.bfgs_line_search, 'backtracking')
                alpha = ArmijoLineSearch(@(x) self.obj(x), objFuncValue(1) ,x_k(:,1), dx(:,1), dir, struct('c', self.c));
            else
                error('incorrect line_method, must be none or backtracking')
            end
            
            p = alpha*dir;
            
            % update x
            x_k(:, 2:end) = x_k(:, 1:end-1);
            x_k(:, 1) = x_k(:, 1) + p;
            s_k = -diff(x_k, [], 2);
            
            self.update_x(x_k(:, 1));
            
            objFuncValue(2) = objFuncValue(1);
            objFuncValue(1) = self.objx;
            
            iter = iter + 1;
            if self.verbose
                fprintf(1,'LBFGS: Iteration %d: alpha_min=%f, OF=%f\n',iter,alpha,objFuncValue(1));
            end
            dx(:,2:end) = dx(:,1:end-1);
            dx(:, 1) = self.gradx;
            y_k = -diff(dx, [], 2);
            
            r_k = 1./diag(y_k'*s_k);
            
            
            % log error measures
            if self.calculate_errors
                self.error_statistics;
            end
        end
        
        if self.verbose
            fprintf(['\n LBFGS:' num2str(iter) ' iteration(s) performed to converge\n'])
        end
    end
      
           
end %methods
end %classdef



















































