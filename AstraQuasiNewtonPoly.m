classdef AstraQuasiNewtonPoly < AstraQuasiNewton
    %ASTRAQUASINEWTONPOLY Handles the quasiNewton methods for ||polyproj(x)-b||_2
    %   Both for the log corrected and non-logcorrected model.
    % Date: 22-11-2021
    % Author: Nathanael Six
    % Affiliation: imec-VisionLab, University of Antwerp, Belgium
    % Contact: nathanael.six@uantwerpen.be
    % License: GPLv3
    
properties
    w       = [] % weights for the different energies
    mu      = [] % attenuation values at the different materials and energies Nm x Ne vector
    Nm      = [] % number of materials
    Ne      = [] % number of energy bins
    ref     = [] % vector Nm x 1 of reference attenuation values
    Nproj   = [] % number of pixels in a projection
    Nvol    = [] % number of pixels in a volume, this is the same as N
    exp_mask_mu = [] % Nproj x Ne in size, might become a storage problem. Code could be rewritten to only store Nproj x Nm elements and even (with more for loops) nothing at all
    deriv_mask = [] % Nvol x Nm in size
    
    % flags
    flag_vaccuum = true % select true if the first material in the basis is vacuum, this is usually true, and in that case the projections with vaccuum don't need to be calculated, as they are 0
    
    % log_corrected options
    is_logcorrected = true
    exp_form = []
    
    % for the convolution of the material triangle functions
    conv_acc = 1e-4
    
  
    
   
end    
    
methods
    function self = AstraQuasiNewtonPoly(A, x0, b, w, mu, ref, options)
        %NewtonKrylovExponential Construct an instance of this class
        %   options is a struct for keyword arguments, can be used to
        %   initialize every other parameter of the class.
        narginchk(6,7);
        if nargin == 6
            options = struct;
        end
        self@AstraQuasiNewton(A, x0, b, options);
        
        self.w = w(:) ./ sum(w(:));
        self.mu = mu;
        if numel(ref) == 1
            self.ref = mu(:, ref);
        else
            self.ref = ref(:);
        end
        
        self.Nm = size(mu, 1);
        self.Ne = size(mu, 2);
        self.Nproj = self.A.m;
        self.Nvol = self.A.n;

        if self.N ~= self.Nvol
            error("Dimensions of x and A don't agree")
        end
        if self.Nm ~= numel(self.ref) || self.Ne ~= numel(self.w)
            error('Dimensions of mu, w and ref are incompatible')
        end
        
        if self.mu(1, :) ~= zeros(size(self.mu(1, :)))
            error('First material in mu has to be vaccuum')
        end
        
        
        
    end %function
    
    function out = mse_gt(self, z)
        % overloading mse_gt function because in the polychromatic model we should
        % do a cutoff of the reconstruction data before calculating the
        % difference
        self.method_counter.mse_gt = self.method_counter.mse_gt + 1;                
        z(z < self.ref(1)) = self.ref(1);
        z(z > self.ref(end)) = self.ref(end);
        out = sum((z(:) - self.gt(:)).^2) / self.N;
    end
    
    function self = update_pre_f(self)
        % updates mask and derived mask 
       self.method_counter.update_pre_f = self.method_counter.update_pre_f + 1;                        
        self.exp_mask_mu = self.calc_mask_lengths(self.x);
        self.deriv_mask = self.calc_deriv_mask(self.x);
    end
    
    function out = f(self, z)
       self.method_counter.f = self.method_counter.f + 1;                
        if nargin == 1
            exp_masks = self.exp_mask_mu;
        else
            exp_masks = self.calc_mask_lengths(z);
        end
        out = exp_masks * self.w;
        if self.is_logcorrected
            if nargin == 1
                self.exp_form = out;
            end
            out = -log(out);
        end
    end

    function Jy = jacobian(self, y, is_transposed)
        % is_transposed = boolean
        % returns Jx*y or Jx'*y, with self.x and self.fx assumed to
        % already be up to date.
        % never used in the method, only to define gradient and Hessian
        % method assumes it always needs to operate on self.x and self.fx

        % this jacobian is the jacobian of the OPERATOR on x
       self.method_counter.jacobian = self.method_counter.jacobian + 1;                        
        if is_transposed
            if self.is_logcorrected
                y = -y ./ self.exp_form;
            end
            if self.flag_vaccuum
            Jy = -sum(self.deriv_mask(:, 2:end) .*  ...
                    (self.A' * ((self.exp_mask_mu .*  ...
                    (y)) * (self.mu(2:end, :)' .* self.w))), 2);
            else
            Jy = -sum(self.deriv_mask .*  ...
                    (self.A' * ((self.exp_mask_mu .*  ...
                    (y)) * (self.mu' .* self.w))), 2);                
            end

        else
            if self.flag_vaccuum
            Jy = -((self.exp_mask_mu .* ...
                    ((self.A * (self.deriv_mask(:, 2:end) .* y)) * self.mu(2:end, :))) ...
                    * self.w);
            else
            Jy = -((self.exp_mask_mu .* ...
                    ((self.A * (self.deriv_mask .* y)) * self.mu)) ...
                    * self.w);                
            end
                
                
            if self.is_logcorrected
                Jy = -Jy ./ self.exp_form;
            end
        end
    end
    
    function exp_masks = calc_mask_lengths(self, z)
       self.method_counter.calc_mask_lengths = self.method_counter.calc_mask_lengths + 1;                        
        masks = zeros(self.Nvol, self.Nm);

        masks(:, 1) = self.conv_bump_triangle(z, self.conv_acc, [], ...
            self.ref(1), self.ref(2));
        masks(:, self.Nm) = self.conv_bump_triangle(z, self.conv_acc, ...
            self.ref(self.Nm - 1), self.ref(self.Nm), []);
        if self.Nm > 2
            for ii = 2:(self.Nm-1)
                masks(:, ii) = self.conv_bump_triangle(z, self.conv_acc, ...
                    self.ref(ii-1), self.ref(ii), self.ref(ii+1));
            end
        end
        
        if self.flag_vaccuum
            exp_masks = exp(-(self.A * masks(:, 2:end)) * self.mu(2:end, :));
        else
            exp_masks = exp(-(self.A * masks) * self.mu);
        end
    end

    function deriv_mask = calc_deriv_mask(self, z)
       self.method_counter.calc_deriv_mask = self.method_counter.calc_deriv_mask + 1;                        
        deriv_mask = zeros(self.Nvol, self.Nm);

        deriv_mask(:, 1) = self.conv_deriv_bump_triangle(z, ...
            self.conv_acc, [], self.ref(1), self.ref(2));
        deriv_mask(:, self.Nm) = self.conv_deriv_bump_triangle(z,...
            self.conv_acc, self.ref(self.Nm - 1), self.ref(self.Nm), []);
        if self.Nm > 2
            for ii = 2:self.Nm-1
                deriv_mask(:, ii) = self.conv_deriv_bump_triangle(...
                    z, self.conv_acc, self.ref(ii-1), self.ref(ii), ...
                    self.ref(ii+1));
            end
        end

    end    
    
    function out = conv_bump_triangle(~, x, epsilon, a, b, c)
        
    out = zeros(size(x));
    
    if numel(a) == 0 && numel(c) ~= 0
        maskminb = x < b-epsilon;
        maskbb = x >= b-epsilon & x < b+epsilon;
        maskbc = x >= b+epsilon & x < c-epsilon;
        maskcc = x >= c-epsilon & x < c+epsilon;
        
        
        out(maskminb) = 1;
        out(maskbb) = 1/4*(4*c*epsilon - epsilon^2 - 2*epsilon*b - b^2 -... 
            2*(epsilon - b)*x(maskbb) - x(maskbb).^2) / ...
            (c*epsilon - epsilon*b);
        
        out(maskbc) = (c - x(maskbc)) / (c - b) ;
        
        out(maskcc) = 1/4*(c^2 + 2*c*epsilon + epsilon^2 - ...
            2*(c + epsilon)*x(maskcc) + x(maskcc).^2) / ...
            (c*epsilon - epsilon*b);
        
        
    elseif numel(c) == 0 && numel(a) ~= 0
        maskaa = x > a-epsilon & x < a+epsilon;
        maskab = x >= a+epsilon & x < b-epsilon;
        maskbb = x >= b-epsilon & x < b+epsilon;
        maskplusb = x >= b+epsilon;
        
        out(maskaa) = -1/4*(a^2 - 2*a*epsilon + epsilon^2 - ...
            2*(a - epsilon)*x(maskaa) + x(maskaa).^2) / ...
            (a*epsilon - epsilon*b);
        
        out(maskab) = (x(maskab) - a) / (b - a);
        
        out(maskbb) = 1/4*(4*a*epsilon + epsilon^2 - 2*epsilon*b + b^2 -...
            2*(epsilon + b)*x(maskbb) + x(maskbb).^2) / ...
            (a*epsilon - epsilon*b);
        
        out(maskplusb) = 1;
        
    else
        maskaa = x > a-epsilon & x < a+epsilon;
        maskab = x >= a+epsilon & x < b-epsilon;
        maskbb = x >= b-epsilon & x < b+epsilon;
        maskbc = x >= b+epsilon & x < c-epsilon;
        maskcc = x >= c-epsilon & x < c+epsilon;
        

        out(maskaa) = -1/4*(a^2 - 2*a*epsilon + epsilon^2 - ...
            2*(a - epsilon)*x(maskaa) + x(maskaa).^2) / ...
            (a*epsilon - epsilon*b);
        
        out(maskab) = (x(maskab) - a) / (b - a);
        
        out(maskbb) = 1/4*(4*a*c*epsilon - (a - c)*epsilon^2 -...
            2*(a + c)*epsilon*b - (a - c)*b^2 - (a - c)*x(maskbb).^2  ...
            - 2*((a + c)*epsilon - (a - c + 2*epsilon)*b)*x(maskbb)) / ...
            (a*c*epsilon - (a + c)*epsilon*b + epsilon*b^2);
        
        out(maskbc) = (c - x(maskbc)) / (c - b) ;
        
        out(maskcc) = 1/4*(c^2 + 2*c*epsilon + epsilon^2 - ...
            2*(c + epsilon)*x(maskcc) + x(maskcc).^2) / ...
            (c*epsilon - epsilon*b);
    end

    end
    
    function out = conv_deriv_bump_triangle(~, x, epsilon, a, b, c)

    out = zeros(size(x));

    if numel(a) == 0 && numel(c) ~= 0
        maskbb = x >= b-epsilon & x < b+epsilon;
        maskbc = x >= b+epsilon & x < c-epsilon;
        maskcc = x >= c-epsilon & x < c+epsilon;
        
        
        out(maskbb) = -1/2*(b - epsilon - x(maskbb))/((b - c)*epsilon);
        
        out(maskbc) = -1 / (c - b) ;
        
        out(maskcc) = 1/2*(c + epsilon - x(maskcc)) / ((b - c)*epsilon);
        
        
    elseif numel(c) == 0 && numel(a) ~= 0
        maskaa = x > a-epsilon & x < a+epsilon;
        maskab = x >= a+epsilon & x < b-epsilon;
        maskbb = x >= b-epsilon & x < b+epsilon;
        
        out(maskaa) = 1/2*(a - epsilon - x(maskaa))/((a - b)*epsilon);
        
        out(maskab) = 1 / (b - a);
        
        out(maskbb) = -1/2*(b + epsilon - x(maskbb))/((a - b)*epsilon);
                
    else
        maskaa = x > a-epsilon & x < a+epsilon;
        maskab = x >= a+epsilon & x < b-epsilon;
        maskbb = x >= b-epsilon & x < b+epsilon;
        maskbc = x >= b+epsilon & x < c-epsilon;
        maskcc = x >= c-epsilon & x < c+epsilon;
        

        out(maskaa) = 1/2*(a - epsilon - x(maskaa))/((a - b)*epsilon);
        
        out(maskab) = 1 / (b - a);
        
        out(maskbb) = -1/2*(a*b - b*c - (a - 2*b + c)*epsilon - ...
            (a - c)*x(maskbb)) / ((a*b - b^2 - (a - b)*c)*epsilon);
        
        out(maskbc) = -1 / (c - b) ;
        
        out(maskcc) = 1/2*(c + epsilon - x(maskcc)) / ((b - c)*epsilon);
    end
    end
    
end %methods
end %classdef

