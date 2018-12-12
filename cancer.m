% matlab -nodesktop -nosplash -nosoftwareopengl < sprinkler.mat

% false = 1; true = 2

addpath('~/asigDSIC/ETSINF/apr/p2/BNT')
addpath(genpathKPM('~/asigDSIC/ETSINF/apr/p2/BNT'))

N = 5;  P = 1; F = 2; C = 3; X = 4; D = 5;
grafo          = zeros(N, N);
grafo(P, C) 	= 1;
grafo(F, C) 	= 1;
grafo(C, [X D]) = 1;

nodosDiscretos = 1:N;
tallaNodos     = [2 2 2 3 2];

redB = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos);

redB.CPD{P} = tabular_CPD(redB, P, [0.9 0.1]);
redB.CPD{F} = tabular_CPD(redB, F, [0.7 0.3]);
redB.CPD{C} = tabular_CPD(redB, C, [0.999 0.95 0.97 0.92 0.001 0.05 0.03 0.08]);
redB.CPD{X} = tabular_CPD(redB, X, [0.8 0.1 0.1 0.7 0.1 0.2]); %n=1 p=2 d=3
redB.CPD{D} = tabular_CPD(redB, D, [0.7 0.35 0.3 0.65]);

motor = jtree_inf_engine(redB);
evidencia = cell(1,N);
evidencia{F} = 1; % no fumador
evidencia{X} = 1; % radiografia negativo
evidencia{D} = 2; % sufre disnea

[explMasProb, logVer] = calc_mpe(motor, evidencia);
explMasProb
logVer

[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor,C,1);
m.T
