addpath('/storage/home/hcoda1/6/sbryngelson3/Code_Riemann_3D_35mom_july2026_GT/src');
OUT='/tmp/kdiff';

% A) full wrapper realizable_3D(M4,Ma)
M4mat = dlmread(fullfile(OUT,'proj_M4.txt'));
N = size(M4mat,1);
for Ma = [2 5]
    R = zeros(N,35);
    for c=1:N
        R(c,:) = realizable_3D(M4mat(c,:), Ma);
    end
    writematrix(R, fullfile(OUT, sprintf('ml_M4r_Ma%d.txt',Ma)), 'Delimiter',' ');
end

% B) projection35 isolation
inp = dlmread(fullfile(OUT,'proj28_in.txt'));
M = size(inp,1);
out = zeros(M,28);
for c=1:M
    a = num2cell(inp(c,:));
    [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19,o20,o21,o22,o23,o24,o25,o26,o27,o28] = projection35(a{:});
    out(c,:) = [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19,o20,o21,o22,o23,o24,o25,o26,o27,o28];
end
writematrix(out, fullfile(OUT,'ml_proj28_out.txt'), 'Delimiter',' ');
disp('MATLAB projection outputs written');
