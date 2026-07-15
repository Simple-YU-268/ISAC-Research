function E = build_E_m(M, Nt)
%BUILD_E_M  Construct per-AP extraction matrices E_m (cell array)
%
%   E{m} is N x N zero everywhere except an identity block on the m-th AP.

N = M * Nt;
E = cell(M, 1);
for m = 1:M
    Em = zeros(N, N);
    Em((m-1)*Nt + 1 : m*Nt, (m-1)*Nt + 1 : m*Nt) = eye(Nt);
    E{m} = Em;
end
end
