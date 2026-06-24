addpath('/storage/home/hcoda1/6/sbryngelson3/Code_Riemann_3D_35mom_july2026_GT/src');
OUT='/tmp/kdiff';

triples = [ ...
 0 0 0;1 0 0;2 0 0;3 0 0;4 0 0; ...
 0 1 0;1 1 0;2 1 0;3 1 0; ...
 0 2 0;1 2 0;2 2 0; ...
 0 3 0;1 3 0; ...
 0 4 0; ...
 0 0 1;1 0 1;2 0 1;3 0 1; ...
 0 0 2;1 0 2;2 0 2; ...
 0 0 3;1 0 3; ...
 0 0 4; ...
 0 1 1;1 1 1;2 1 1; ...
 0 2 1;1 2 1; ...
 0 3 1; ...
 0 1 2;1 1 2; ...
 0 1 3; ...
 0 2 2];

M4mat = dlmread(fullfile(OUT,'M4.txt'));
NCASE = size(M4mat,1);
C4o=zeros(NCASE,35); S4o=zeros(NCASE,35); Eo=zeros(NCASE,36);
hyqo=zeros(NCASE,21); rt=zeros(NCASE,35);

sidx = [4 5 7 8 9 11 12 13 14 15 17 18 19 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35];

for c=1:NCASE
    M4 = M4mat(c,:);
    [C4,S4] = M2CS4_35(M4);
    C4o(c,:)=C4; S4o(c,:)=S4;
    a = num2cell(S4(sidx));
    E = delta2star3D(a{:});
    Eo(c,:) = E(:)';
    [S500,S410,S320,S230,S140,S401,S302,S203,S104,S311,S221,S131,S212,S113,S122,S050,S041,S032,S023,S014,S005] = hyqmom_3D(a{:});
    hyqo(c,:) = [S500,S410,S320,S230,S140,S401,S302,S203,S104,S311,S221,S131,S212,S113,S122,S050,S041,S032,S023,S014,S005];
    % round trip
    M000=M4(1); um=M4(2)/M000; vm=M4(6)/M000; wm=M4(16)/M000;
    C=C4;
    M5 = C4toM4_3D(M000,um,vm,wm, C(3),C(7),C(17),C(10),C(26),C(20), ...
        C(4),C(8),C(18),C(11),C(27),C(21),C(13),C(29),C(32),C(23), ...
        C(5),C(9),C(19),C(12),C(28),C(22),C(14),C(30),C(33),C(24),C(15),C(31),C(35),C(34),C(25));
    for n=1:35
        rt(c,n) = M5(triples(n,1)+1, triples(n,2)+1, triples(n,3)+1);
    end
end
writematrix(C4o, fullfile(OUT,'ml_C4.txt'),'Delimiter',' ');
writematrix(S4o, fullfile(OUT,'ml_S4.txt'),'Delimiter',' ');
writematrix(Eo,  fullfile(OUT,'ml_E.txt'),'Delimiter',' ');
writematrix(hyqo,fullfile(OUT,'ml_hyq.txt'),'Delimiter',' ');
writematrix(rt,  fullfile(OUT,'ml_roundtrip.txt'),'Delimiter',' ');

% realizability_S2
s2in = dlmread(fullfile(OUT,'s2_in.txt'));
s2out = zeros(size(s2in,1),4);
for c=1:size(s2in,1)
    [a,b,d,e] = realizability_S2(s2in(c,1),s2in(c,2),s2in(c,3));
    s2out(c,:) = [a b d e];
end
writematrix(s2out, fullfile(OUT,'ml_s2_out.txt'),'Delimiter',' ');

% realizablity_S220
s220in = dlmread(fullfile(OUT,'s220_in.txt'));
s220out=zeros(size(s220in,1),1);
for c=1:size(s220in,1)
    s220out(c)=realizablity_S220(s220in(c,1),s220in(c,2),s220in(c,3));
end
writematrix(s220out, fullfile(OUT,'ml_s220_out.txt'),'Delimiter',' ');
disp('MATLAB outputs written');
