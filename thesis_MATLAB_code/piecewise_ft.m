% Gaussian -> flat top fit
function y = piecewise_ft(xL, xR, a, b, d, e, g, h, k, x)
y = zeros(size(x));

%xL = -0.015;
%xR =  0.015;

polyL = g*xL^2 + h*xL + k;
polyR = g*xR^2 + h*xR + k;

% Solve c, f so the exponentials meet the polynomial exactly at breakpoints
c = log(polyL/a) - b*xL;
f = log(polyR/d) - e*xR;
for ii = 1:length(x)
    if (x(ii) < xL) % Exponential to edge of cathode
        y(ii) = a.*exp(b.*x(ii)+c);
    elseif(x(ii) > xR) % Exponential after edge of cathode
        y(ii) = d.*exp(e.*x(ii)+f);
    else % Poly2 in the middle of cathode (goes from gaussian to flat)
        y(ii) = g.*x(ii)^2+h.*x(ii)+k;
    end
end
end
