% 同步辐射X射线衍射能量反推程序
% 基于粉末标样的衍射环半径比值求解X射线能量

clear; clc;
%% 1. 输入标样信息（以LaB6为例）
% % LaB6立方晶系，空间群Pm-3m
% a = 4.15695;  % 晶格常数 (Å)

% % 定义可能出现的衍射峰 (h k l)
% hkl = [
%     1 0 0;
%     1 1 0;
%     1 1 1;
%     2 0 0;
%     2 1 0;
%     2 1 1;
%     2 2 0;
%     3 0 0;
%     3 1 0;
%     3 1 1;
%     2 2 2;
%     3 2 0;
%     3 2 1;
%     4 0 0;
%     4 1 0;
%     4 1 1;
%     3 3 1;
%     4 2 0;
%     4 2 1;
%     3 3 2;
%     4 2 2;
%     5 0 0;
%     5 1 0;
%     5 1 1;
%     5 2 0;
%     5 2 1;
% ];

% CeO2
a = 5.411651;  % 晶格常数 (Å)5.411651
% 定义可能出现的衍射峰 (h k l)
hkl = [
    1, 1, 1;
    2, 0, 0;
    2, 2, 0;
    3, 1, 1;
    2, 2, 2;
    4, 0, 0;
    3, 3, 1;
    4, 2, 0;
    4, 2, 2;
    5, 1, 1;
    4, 4, 0;
    5, 3, 1;
    6, 0, 0;
    6, 2, 0;
    5, 3, 3;
    6, 2, 2;
    4, 4, 4;
    7, 1, 1;
    6, 4, 0;
    6, 4, 2;
    7, 3, 1;
    8, 0, 0;
    7, 3, 3;
    8, 2, 0;
    8, 2, 2;
    7, 5, 1;
    6, 6, 2;
    8, 4, 0;
    9, 1, 1;
    8, 4, 2;
    6, 6, 4;
    9, 3, 1;
    8, 4, 4;
    9, 3, 3;
    10, 0, 0;
    10, 2, 0;
    9, 5, 1;
    10, 2, 2;
    9, 5, 3;
    10, 4, 0;
    10, 4, 1;
    ];

% 计算各衍射峰对应的d间距
d_hkl = a ./ sqrt(hkl(:,1).^2 + hkl(:,2).^2 + hkl(:,3).^2);


%% 2. 实测 R ratio
lambda_ini = 0.123984; % in Å, 初步估计波长
d_measured = [3.12419475103949	2.70564815818238	1.91324849714668	1.63166325530814	1.56222318955555	1.35297791717193];
ttheta_measured = 2*asin(lambda_ini./2./d_measured); % in rad
R_ratio = tan(ttheta_measured) ./ tan(ttheta_measured(1)); % 实测环半径比

% 指定实测的衍射峰索引（需要根据强度和经验判断）
peak_indices = 1:numel(d_measured);     % 对应hkl列表中的索引
d_used = d_hkl(peak_indices);       % in Å
fprintf('使用的衍射峰 (hkl) 和对应的d间距:\n');
for i = 1:length(peak_indices)
    fprintf('Peak %d: (%d %d %d), d = %.4f Å\n', i, ...
        hkl(peak_indices(i),1), hkl(peak_indices(i),2), hkl(peak_indices(i),3), ...
        d_used(i));
end

%% 3. 最优化能量
[E_keV, lambda_A, res] = estimate_energy_from_R(R_ratio, d_used);
fprintf('Estimated\t E = %.4f keV, lambda = %.6f Å\n', E_keV, lambda_A);

%% Functions
function [E_keV, lambda_A, res] = estimate_energy_from_R(R_ratio, dA)
% estimate_energy_from_R  根据Debye环半径比值估计X射线能量（及可选D）
%   R_ratio   : Nx1 vector of measured ring radii ratio.
%   dA  : Nx1 vector of corresponding d-spacings in Angstroms (Å)
% Returns:
%   E_keV   : estimated photon energy in keV
%   lambda_A: estimated wavelength in Å
%   res     : optimization residual (sum squares of ratio differences)
%
% Usage:
%   [E, lambda, D, res] = estimate_energy_from_R(R, d);
%
% Notes:
%   - Need at least two rings.
%   - The function uses ratio-based least-squares to eliminate D and find lambda.
%   - It is robust to scale of R because ratios are used.

% check inputs
R_ratio = R_ratio(:);
dA = dA(:);
if numel(R_ratio) ~= numel(dA)
    error('R_ratio and dA must have same length');
end
N = numel(dA);
if N < 2
    error('Need at least two rings to solve for energy.');
end

% objective function: sum of squared differences between measured ratios and model ratios
% unknown: lambda (Å)
obj = @(lambda) ratio_residual(lambda, R_ratio, dA);

% bounds: lambda in (small positive, 2*min(dA)*0.999)
lambda_min = 1e-6;
lambda_max = 2 * min(dA) * 0.999;

% initial guess: use approximate Bragg for first two rings at small angles
% % approximate theta ~ atan(R/D) unknown D => use small-angle approx with ratio => initial guess = 12.4/Eguess
% lambda0 = min( (2*dA(1)) * 0.5, (2*dA(2)) * 0.5 );
% better: start mid of allowed range
% lambda0 = 0.5*(lambda_min + lambda_max);

% use fminsearch with bounds via transform, or fminbnd if available
opts = optimset('TolX',1e-8,'TolFun',1e-10,'Display','off','MaxFunEvals',5000,'MaxIter',5000);
% use fminbnd
lambda_opt = fminbnd(obj, lambda_min, lambda_max, opts);

% compute final values
lambda_A = lambda_opt;

% compute energy
hc = 4.135667696*2.99792458;   
E_keV = hc ./ lambda_A; % keV (λ in Å)

% residual
res = obj(lambda_A);

end

function s = ratio_residual(lambda, R_ratio, dA)
    % compute model ratios relative to first ring
    theta = lambda./(2*dA);
    % discard invalid (sin>1)
    if any(theta>=1)
        s = 1e6 + sum((theta(theta>=1)-1).^2).*1e6;
        return;
    end
    theta = asin(theta); % radians
    model = tan(2*theta);
    model_rat = model ./ model(1);
    data_rat = R_ratio;
    % residual (weighted)
    err = data_rat - model_rat;
    s = sum(err.^2);
end